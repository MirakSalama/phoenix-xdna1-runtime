#include "phoenix_prefill.h"
#include "ggml.h"

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
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t output_width = 2560;
constexpr size_t input_width = 960;
constexpr size_t block = 128;

std::filesystem::path program_path() {
    static const char address = 0;
    HMODULE module{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&address), &module)) throw std::runtime_error("Prefill module unavailable");
    std::wstring path(32768, 0);
    const auto n = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (!n || n >= path.size()) throw std::runtime_error("Prefill module path unavailable");
    path.resize(n);
    return std::filesystem::path(path).parent_path() / "programs" / "prefill-2560x960x128";
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
    xrt::bo weights;
};

struct PhoenixPrefill::Impl {
    xrt::device device{0};
    xrt::xclbin image;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instructions, x, y;
    xrt::run run;
    ggml_bf16_t* input;
    float* result;

    Impl() {
        const auto name = device.get_info<xrt::info::device::name>();
        if (name.find("Phoenix") == std::string::npos && name.find("Hawk") == std::string::npos)
            throw std::runtime_error("Unsupported Phoenix prefill device");
        const auto stem = program_path();
        image = xrt::xclbin(read_file(stem, L".xclbin"));
        device.register_xclbin(image);
        context = xrt::hw_context(device, image.get_uuid());
        kernel = xrt::kernel(context, "MLIR_AIE");
        const auto code = read_file(stem, L".bin");
        if (code.size() % 4) throw std::runtime_error("Invalid Phoenix prefill instructions");
        instructions = xrt::bo(device, code.size(), xrt::bo::flags::cacheable, kernel.group_id(1));
        instructions.write(code.data(), code.size(), 0);
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        x = xrt::bo(device, block*input_width*sizeof(ggml_bf16_t), xrt::bo::flags::host_only, 0);
        y = xrt::bo(device, block*output_width*sizeof(float), xrt::bo::flags::host_only, 0);
        input = x.map<ggml_bf16_t*>();
        result = y.map<float*>();
        run = xrt::run(kernel);
        run.set_arg(0, uint32_t{3});
        run.set_arg(1, instructions);
        run.set_arg(2, static_cast<uint32_t>(code.size()));
        run.set_arg(4, x);
        run.set_arg(5, y);
    }
};

PhoenixPrefill::PhoenixPrefill() : context_(std::make_unique<Impl>()) {}
PhoenixPrefill::~PhoenixPrefill() = default;

void PhoenixPrefill::set_prepare_threads(int threads) {
    prepare_threads_ = std::clamp(threads, 1, 4);
}

bool PhoenixPrefill::supports(size_t rows, size_t columns, size_t tokens) {
    return rows == output_width && columns == input_width && tokens >= 96 && tokens <= 4*block;
}

size_t PhoenixPrefill::packed_bytes(size_t rows, size_t columns) {
    if (!supports(rows, columns, block)) throw std::invalid_argument("Unsupported Phoenix prefill weight shape");
    return rows*columns*sizeof(uint16_t);
}

std::shared_ptr<const PhoenixPrefill::Prepared> PhoenixPrefill::prepare(
    size_t rows, size_t columns, PhoenixMatvec::RowDecoder decode, void* source) {
    if (!decode) throw std::invalid_argument("Missing Phoenix prefill weight decoder");
    auto p = std::make_shared<Prepared>();
    p->weights = xrt::bo(context_->device, packed_bytes(rows,columns), xrt::bo::flags::host_only, 0);
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

void PhoenixPrefill::compute(const Prepared& weights, const float* activations, float* out, size_t tokens) {
    if (!activations || !out || !supports(output_width,input_width,tokens))
        throw std::invalid_argument("Unsupported Phoenix prefill projection");
    auto & ctx = *context_;
    ctx.run.set_arg(3,weights.weights);
    for (size_t t=0; t<tokens; t+=block) {
        const size_t active = std::min(block,tokens-t);
        ggml_fp32_to_bf16_row(activations+t*input_width,ctx.input,active*input_width);
        std::fill(ctx.input+active*input_width,ctx.input+block*input_width,ggml_bf16_t{});
        ctx.x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        ctx.run.start();
        const auto state = ctx.run.wait(std::chrono::seconds(10));
        if (state == ERT_CMD_STATE_TIMEOUT) ctx.run.abort();
        if (state != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("Phoenix prefill failed, XRT command state " +
                                     std::to_string(static_cast<int>(state)));
        const size_t elements = active*output_width;
        ctx.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE,elements*sizeof(float),0);
        std::copy_n(ctx.result,elements,out+t*output_width);
    }
}
