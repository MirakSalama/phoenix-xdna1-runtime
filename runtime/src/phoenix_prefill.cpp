#include "phoenix_prefill.h"
#include "phoenix_q4_prompt.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <immintrin.h>
#include <list>
#include <tuple>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t block = 128;
constexpr size_t input_slots = 4;

std::filesystem::path program_path(size_t rows, size_t columns, size_t batch, bool resident) {
    static const char address = 0;
    HMODULE module{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&address), &module)) throw std::runtime_error("Prefill module unavailable");
    std::wstring path(32768, 0);
    const auto n = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (!n || n >= path.size()) throw std::runtime_error("Prefill module path unavailable");
    path.resize(n);
    return std::filesystem::path(path).parent_path() / "programs" /
           ((resident ? "resident-" : "prefill-")+std::to_string(rows)+"x"+std::to_string(columns)+"x"+std::to_string(batch));
}

std::vector<char> read_file(std::filesystem::path path, const wchar_t* extension) {
    path += extension;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = input.tellg();
    if (!input || bytes <= 0 || bytes > (1<<24)) throw std::runtime_error("Invalid Phoenix prefill artifact");
    std::vector<char> result(static_cast<size_t>(bytes));
    input.seekg(0);
    if (!input.read(result.data(), bytes)) throw std::runtime_error("Incomplete Phoenix prefill artifact");
    return result;
}
}

struct PhoenixPrefill::Prepared {
    size_t rows, columns;
    bool resident = false;
    xrt::bo weights;
};

bool PhoenixPrefill::is_resident_bf16(const Prepared& weights) { return weights.resident; }

bool PhoenixPrefill::resident_shape(size_t rows, size_t cols) {
    return (rows == 7680 && cols == 2048) || ((rows == 6144 || rows == 12288) && cols == 1536) ||
           (rows == 1536 && cols == 6144);
}

namespace {
// Reproduce the stock float32 -> BF16 round-to-nearest-even on finite input.
void store_bf16(__m256 values, void* dst) {
    const auto bits = _mm256_castps_si256(values);
    const auto upper = _mm256_srli_epi32(bits,16);
    const auto bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(upper,_mm256_set1_epi32(1)));
    const auto rounded = _mm256_srli_epi32(_mm256_add_epi32(bits,bias),16);
    _mm_storeu_si128(static_cast<__m128i*>(dst),
                     _mm_packus_epi32(_mm256_castsi256_si128(rounded),_mm256_extracti128_si256(rounded,1)));
}

void stage_q8_0(const block_q8_0* src, ggml_bf16_t* dst, size_t count) {
    for (size_t offset=0; offset<count; offset+=QK8_0, ++src) {
        if ((src->d & 0x7c00) == 0x7c00) {
            float values[QK8_0];
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(src, values, QK8_0);
            ggml_fp32_to_bf16_row(values,dst+offset,QK8_0);
            continue;
        }
        const auto scale = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(src->d)));
        for (size_t i=0; i<QK8_0; i+=8) {
            const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src->qs+i));
            const auto q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            store_bf16(_mm256_mul_ps(q,scale),dst+offset+i);
        }
    }
}

struct ProjectionProgram {
    xrt::device& device;
    const size_t input_width;
    const size_t batch;
    xrt::xclbin image;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instructions, y;
    std::array<xrt::bo, input_slots> inputs;
    xrt::run run;
    float* result;
    std::vector<float> previous_input, rounded;
    std::vector<uint8_t> codes;
    ggml_type staged_type = GGML_TYPE_COUNT;

    ProjectionProgram(xrt::device& dev, size_t rows, size_t columns, size_t batch_width, bool resident)
        : device(dev), input_width(columns), batch(batch_width) {
        const auto stem = program_path(rows, columns, batch, resident);
        image = xrt::xclbin(read_file(stem, L".xclbin"));
        device.register_xclbin(image);
        context = xrt::hw_context(device, image.get_uuid());
        kernel = xrt::kernel(context, "MLIR_AIE");
        const auto code = read_file(stem, L".bin");
        if (code.size() % 4) throw std::runtime_error("Invalid Phoenix prefill instructions");
        instructions = xrt::bo(device, code.size(), xrt::bo::flags::cacheable, kernel.group_id(1));
        instructions.write(code.data(), code.size(), 0);
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        y = xrt::bo(device, batch*rows*sizeof(float), xrt::bo::flags::host_only, 0);
        result = y.map<float*>();
        run = xrt::run(kernel);
        run.set_arg(0, uint32_t{3});
        run.set_arg(1, instructions);
        run.set_arg(2, static_cast<uint32_t>(code.size()));
        run.set_arg(5, y);
    }
    bool matches(const float* activations, size_t count, ggml_type dot) const {
        return staged_type == dot && previous_input.size() == count &&
               std::memcmp(previous_input.data(), activations, count * sizeof(float)) == 0;
    }

    void upload_tile(size_t t, size_t tokens, ggml_type dot) {
        const size_t active = std::min(batch,tokens-t) * input_width;
        auto& bo = inputs[t/batch];
        if (!bo) bo = xrt::bo(device, batch*input_width*sizeof(ggml_bf16_t), xrt::bo::flags::host_only, 0);
        auto* data = bo.map<ggml_bf16_t*>();
        const float* src = previous_input.data() + t * input_width;
        if (dot == GGML_TYPE_Q8_0 || dot == GGML_TYPE_Q8_K) {
            codes.resize(ggml_row_size(dot, active));
            ggml_get_type_traits_cpu(dot)->from_float(src, codes.data(), active);
            if (dot == GGML_TYPE_Q8_K) {
                rounded.resize(active);
                dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(codes.data()), rounded.data(), active);
                ggml_fp32_to_bf16_row(rounded.data(),data,active);
            } else {
                stage_q8_0(reinterpret_cast<const block_q8_0*>(codes.data()),data,active);
            }
        } else {
            ggml_fp32_to_bf16_row(src,data,active);
        }
        std::fill(data+active,data+batch*input_width,ggml_bf16_t{});
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

// Restore the fifth bit and signed offset, multiply by the stored binary16 scale,
// then round each float to nearest-even BF16 exactly like the stock GGML converter.
inline void unpack_q5_eight(__m128i nibble, uint32_t high, __m256 scale, uint16_t* target) {
    const __m256i positions = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i sign_bits = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(high),positions),one);
    const __m256i q = _mm256_add_epi32(_mm256_cvtepu8_epi32(nibble),
                                        _mm256_slli_epi32(_mm256_sub_epi32(sign_bits,one),4));
    store_bf16(_mm256_mul_ps(_mm256_cvtepi32_ps(q),scale),target);
}
}

struct PhoenixPrefill::Impl {
    xrt::device device{0};
    struct CachedProgram {
        std::tuple<size_t, size_t, size_t, bool> shape;
        std::unique_ptr<ProjectionProgram> program;
    };
    std::list<CachedProgram> programs;

    Impl() {
        const auto name = device.get_info<xrt::info::device::name>();
        if (name.find("Phoenix") == std::string::npos && name.find("Hawk") == std::string::npos)
            throw std::runtime_error("Unsupported Phoenix prefill device");
    }
    ProjectionProgram& program(size_t rows, size_t cols, size_t batch, bool resident) {
        const auto shape = std::make_tuple(rows, cols, batch, resident);
        auto found = std::find_if(programs.begin(), programs.end(),
            [&](const CachedProgram& entry) { return entry.shape == shape; });
        if (found != programs.end()) {
            programs.splice(programs.begin(), programs, found);
        } else {
            // Release the oldest hardware context before requesting another from XRT.
            constexpr size_t max_programs = 3;
            if (programs.size() == max_programs)
                programs.pop_back();
            auto program = std::make_unique<ProjectionProgram>(device, rows, cols, batch, resident);
            programs.push_front({shape, std::move(program)});
        }
        return *programs.front().program;
    }
};

void PhoenixPrefill::decode_q5_0_row(const void* packed, uint16_t* dst, size_t columns) {
    if (columns % QK5_0) throw std::invalid_argument("Incomplete Q5_0 row");
    const auto* q = static_cast<const block_q5_0*>(packed);
    for (size_t i = 0; i < columns; i += QK5_0, ++q) {
        if ((q->d & 0x7c00) == 0x7c00) {
            float values[QK5_0];
            ggml_get_type_traits(GGML_TYPE_Q5_0)->to_float(q,values,QK5_0);
            ggml_fp32_to_bf16_row(values,reinterpret_cast<ggml_bf16_t*>(dst+i),QK5_0);
            continue;
        }
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q->qs));
        const __m128i mask = _mm_set1_epi8(15);
        const __m128i low = _mm_and_si128(bytes,mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes,4),mask);
        uint32_t fifth;
        std::memcpy(&fifth,q->qh,sizeof(fifth));
        const __m256 scale = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(q->d)));
        unpack_q5_eight(low,fifth,scale,dst+i);
        unpack_q5_eight(_mm_srli_si128(low,8),fifth>>8,scale,dst+i+8);
        unpack_q5_eight(high,fifth>>16,scale,dst+i+16);
        unpack_q5_eight(_mm_srli_si128(high,8),fifth>>24,scale,dst+i+24);
    }
}

PhoenixPrefill::PhoenixPrefill() : context_(std::make_unique<Impl>()) {}
PhoenixPrefill::~PhoenixPrefill() = default;

void PhoenixPrefill::set_prepare_threads(int threads) {
    prepare_threads_ = std::clamp(threads, 1, 4);
}

bool PhoenixPrefill::supports(size_t rows, size_t columns, size_t tokens) {
    if (tokens > input_slots*block) return false;
    if (rows == 2560 && columns == 960) return tokens >= 64;
    return tokens >= block && ((rows == 7680 && columns == 2048) ||
                              ((rows == 6144 || rows == 12288) && columns == 1536) ||
                              (rows == 1536 && columns == 6144) ||
                              (rows == 2048 && columns == 7680));
}

size_t PhoenixPrefill::packed_bytes(size_t rows, size_t columns) {
    if (!supports(rows, columns, block)) throw std::invalid_argument("Unsupported Phoenix prefill weight shape");
    return rows == 2560 ? rows*columns*sizeof(uint16_t) : (rows*columns/32)*20;
}

std::shared_ptr<const PhoenixPrefill::Prepared> PhoenixPrefill::prepare_q4(
    size_t rows, size_t columns, const void* data, bool cpu_repacked) {
    if (!data || rows == 2560 || !supports(rows, columns, block))
        throw std::invalid_argument("Unsupported compact Phoenix prefill shape");
    auto p = std::make_shared<Prepared>();
    p->rows = rows;
    p->columns = columns;
    p->weights = xrt::bo(context_->device, packed_bytes(rows,columns), xrt::bo::flags::host_only, 0);
    auto* target = p->weights.map<uint8_t*>();
    const int threads = prepare_threads_;
    auto range = [&](int worker) {
        const size_t tiles = rows / 64;
        phoenix_pack_q4_prompt_tiles(static_cast<const uint8_t*>(data), cpu_repacked, columns,
                                     tiles * worker / threads, tiles * (worker + 1) / threads, target);
    };
    std::vector<std::future<void>> jobs;
    for (int i=1; i<threads; ++i) jobs.emplace_back(std::async(std::launch::async, range, i));
    range(0);
    for (auto& job : jobs) job.get();
    p->weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return p;
}

std::shared_ptr<const PhoenixPrefill::Prepared> PhoenixPrefill::prepare(
    size_t rows, size_t columns, PhoenixMatvec::RowDecoder decode, void* source) {
    if (!decode || (!resident_shape(rows,columns) && rows != 2560) || !supports(rows, columns, block))
        throw std::invalid_argument("Unsupported Phoenix BF16 prefill preparation");
    auto p = std::make_shared<Prepared>();
    p->rows = rows;
    p->columns = columns;
    p->resident = rows != 2560;
    p->weights = xrt::bo(context_->device, rows*columns*sizeof(uint16_t), xrt::bo::flags::host_only, 0);
    auto * w = p->weights.map<uint16_t*>();
    const int threads = prepare_threads_;
    auto range = [&](int worker) {
        const size_t past = rows * (worker + 1) / threads;
        for (size_t r = rows * worker / threads; r < past; ++r) decode(source, r, w + r * columns);
    };
    // Each task owns its output rows. Futures join on exceptional exits, before
    // the XRT buffer can be destroyed, and get propagates a worker's error.
    std::vector<std::future<void>> jobs;
    for (int i = 1; i < threads; ++i) jobs.emplace_back(std::async(std::launch::async, range, i));
    range(0);
    for (auto & job : jobs) job.get();
    p->weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return p;
}

bool PhoenixPrefill::compute(const Prepared& weights, const float* activations, float* out, size_t tokens,
                             ggml_type weight_type) {
    if (!activations || !out || tokens == 0 || tokens > input_slots*block)
        throw std::invalid_argument("Unsupported Phoenix prefill projection");
    // A doubled tile only makes sense when the existing program already needs
    // two or four tiles. Keep its 128-token path for the 1/3-tile cases.
    const auto short_tiles = (tokens+block-1)/block;
    const size_t batch = weights.rows != 2560 && short_tiles%2 == 0 ? 256 : block;
    auto & ctx = context_->program(weights.rows, weights.columns, batch, weights.resident);
    const auto dot = ggml_get_type_traits_cpu(weight_type)->vec_dot_type;
    const auto count = tokens * weights.columns;
    const bool reused = ctx.matches(activations,count,dot);
    if (!reused) {
        ctx.staged_type = GGML_TYPE_COUNT;
        ctx.previous_input.assign(activations, activations + count);
        ctx.upload_tile(0,tokens,dot);
    }
    ctx.run.set_arg(3,weights.weights);
    for (size_t t=0; t<tokens; t+=batch) {
        const size_t active = std::min(batch,tokens-t);
        ctx.run.set_arg(4,ctx.inputs[t/batch]);
        ctx.run.start();
        ert_cmd_state state;
        try {
            // The running command reads inputs[t/batch]. Each subsequent tile
            // has its own BO, including padding on an incomplete last tile.
            if (!reused && t+batch < tokens) ctx.upload_tile(t+batch,tokens,dot);
            state = ctx.run.wait(std::chrono::seconds(10));
            if (state == ERT_CMD_STATE_TIMEOUT) throw std::runtime_error("Phoenix prefill timed out");
        } catch (...) {
            // XRT abort is synchronous. If cancellation itself fails, join the
            // command before this stream can reuse its inputs or output.
            try { ctx.run.abort(); } catch (...) { ctx.run.wait(); }
            throw;
        }
        if (state != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("Phoenix prefill failed, XRT command state " +
                                     std::to_string(static_cast<int>(state)));
        const size_t elements = active*weights.rows;
        ctx.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE,elements*sizeof(float),0);
        std::copy_n(ctx.result,elements,out+t*weights.rows);
    }
    ctx.staged_type = dot;
    return reused;
}
