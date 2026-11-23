// Phoenix XDNA 1 is a synchronous ggml accelerator alongside the CPU backend.
// ggml owns graph placement, GGUF loading, tokenization and unsupported ops.
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "phoenix_matvec.h"
#include "phoenix_prefill.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <windows.h>

namespace {

constexpr char NAME[] = "XDNA1";
constexpr char DESCRIPTION[] = "AMD XDNA 1 NPU (Phoenix BF16 matrix-vector)";
ggml_guid guid = {0x19, 0xcf, 0x36, 0x8b, 0x30, 0x7d, 0x40, 0xe1,
                  0xa4, 0xd6, 0xc5, 0xe1, 0x35, 0x59, 0x24, 0x67};
std::atomic<unsigned long long> completed{0};

void trace(const char * line) {
    std::fputs(line, stderr);
    std::fflush(stderr);
    const wchar_t * path = _wgetenv(L"PHOENIX_XDNA_TRACE");
    if (path && *path) {
        FILE * log = nullptr;
        if (_wfopen_s(&log, path, L"ab") == 0) {
            std::fputs(line, log);
            std::fclose(log);
        }
    }
}

bool metadata_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum class Placement { dense, all, head, split_head };

Placement placement() {
    static const Placement configured = [] {
        const char * name = std::getenv("PHOENIX_XDNA_PLACEMENT");
        if (name && std::strcmp(name, "all") == 0) return Placement::all;
        if (name && std::strcmp(name, "head") == 0) return Placement::head;
        if (name && std::strcmp(name, "dense") == 0) return Placement::dense;
        return Placement::split_head;
    }();
    return configured;
}

bool head_mode() {
    return placement() == Placement::head || placement() == Placement::split_head;
}

bool prefill_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("PHOENIX_XDNA_PREFILL");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

size_t prompt_budget() {
    static const size_t bytes = [] {
        const char * value = std::getenv("PHOENIX_XDNA_PROMPT_CACHE_MIB");
        return size_t(value && !std::strcmp(value, "384") ? 384 : 700) << 20;
    }();
    return bytes;
}

bool available_memory(uint64_t free_bytes) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) && status.ullTotalPhys >= (uint64_t(16) << 30) &&
           status.ullAvailPhys >= free_bytes;
}

size_t large_prompt_budget(size_t rows, size_t cols) {
    if (!((rows == 7680 && cols == 2048) || (rows == 2048 && cols == 7680))) return prompt_budget();
    static const size_t selected = [] {
        const char * value = std::getenv("PHOENIX_XDNA_PROMPT_CACHE_MIB");
        if (value && (!std::strcmp(value, "384") || !std::strcmp(value, "700"))) return prompt_budget();
        return available_memory(uint64_t(6) << 30) ? size_t(1320) << 20 : prompt_budget();
    }();
    return selected;
}

bool large_prompt_weight(const ggml_tensor & weight) {
    static const bool enabled = [] {
        const char * value = std::getenv("PHOENIX_XDNA_LARGE_PREFILL");
        return !value || std::strcmp(value, "0") != 0;
    }();
    if (!enabled || weight.type != GGML_TYPE_Q4_0 || std::strncmp(weight.name, "blk.", 4)) return false;
    const char * end = weight.name + std::strlen(weight.name);
    unsigned layer{};
    const auto parsed = std::from_chars(weight.name + 4, end, layer);
    if (parsed.ec != std::errc{}) return false;
    const size_t cols = weight.ne[0], rows = weight.ne[1];
    if (((rows == 7680 && cols == 2048) || (rows == 2048 && cols == 7680)) &&
        large_prompt_budget(rows, cols) > prompt_budget()) {
        const size_t matrix = ggml_row_size(GGML_TYPE_Q4_0, cols) * rows + PhoenixPrefill::packed_bytes(rows, cols);
        const bool down = !std::strcmp(parsed.ptr, ".ffn_down.weight");
        const bool forward = !std::strcmp(parsed.ptr, ".ffn_gate.weight") || !std::strcmp(parsed.ptr, ".ffn_up.weight");
        return ((down && rows == 2048) || (forward && rows == 7680)) && layer < 24 &&
               layer < large_prompt_budget(rows, cols) / (3 * matrix);
    }
    if (!std::strcmp(parsed.ptr, ".ffn_down.weight")) {
        if (cols != 6144 || rows != 1536) return false;
        const size_t matrix = ggml_row_size(GGML_TYPE_Q4_0, cols) * rows + PhoenixPrefill::packed_bytes(rows, cols);
        // Retain room for both gate/up projections in Gemma's 15 narrow
        // blocks. The 384 MiB profile can additionally fit five down tensors.
        const size_t reserved = 2 * 15 * matrix;
        return reserved <= prompt_budget() && layer < 15 && layer < (prompt_budget() - reserved) / matrix;
    }
    if (std::strcmp(parsed.ptr, ".ffn_gate.weight") && std::strcmp(parsed.ptr, ".ffn_up.weight")) return false;
    if (!(cols == 1536 && (rows == 6144 || rows == 12288)) &&
        !(cols == 2048 && rows == 7680)) return false;
    const auto layer_cost = [cols](size_t count) {
        return 2 * (ggml_row_size(GGML_TYPE_Q4_0, cols) * count + PhoenixPrefill::packed_bytes(count, cols));
    };
    // The tested Gemma topology uses 15 narrow FFNs before its wider blocks.
    // Bound this prefix at graph placement, then check actual cache entries
    // before allocation as well, so other GGUF layer arrangements stay bounded.
    if (rows == 12288 && layer >= 15) {
        const size_t first = 15 * layer_cost(6144);
        return first <= prompt_budget() && layer - 15 < (prompt_budget() - first) / layer_cost(rows);
    }
    return layer < prompt_budget() / layer_cost(rows);
}

ggml_backend_buffer_type_t cpu_repack_type() {
    static const auto type = []() -> ggml_backend_buffer_type_t {
        if (!ggml_cpu_has_avx2()) return nullptr;
        const auto reg = ggml_backend_cpu_reg();
        const auto extras = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
        if (!extras) return nullptr;
        for (auto p = extras(ggml_backend_reg_dev_get(reg, 0)); p && *p; ++p) {
            if (!std::strcmp(ggml_backend_buft_name(*p), "CPU_REPACK")) return *p;
        }
        return nullptr;
    }();
    return type;
}

bool output_weight(const ggml_tensor * weight) {
    return !std::strcmp(weight->name, "output.weight") || !std::strcmp(weight->name, "token_embd.weight");
}

struct WeightVersion {
    uint64_t allocation = 0;
    uint64_t changes = 0;
};

struct WatchedBuffer {
    void * data = nullptr;
    std::mutex mutex;
    std::vector<void *> pages;
    std::vector<uint64_t> page_changes;
    size_t page_size = 0;
    WeightVersion version;

    ~WatchedBuffer() {
        if (data) VirtualFree(data, 0, MEM_RELEASE);
    }
};

void free_watched(ggml_backend_buffer_t buffer) {
    delete static_cast<WatchedBuffer *>(buffer->context);
}

ggml_backend_buffer_t allocate_host_weights(ggml_backend_buffer_type_t type, size_t bytes) {
    static std::atomic<uint64_t> next_id{0};
    std::unique_ptr<WatchedBuffer> owner(new (std::nothrow) WatchedBuffer);
    if (!owner) return nullptr;
    owner->data = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
    if (!owner->data) return ggml_backend_cpu_buffer_type()->iface.alloc_buffer(type, bytes);
    auto * buffer = ggml_backend_cpu_buffer_from_ptr(owner->data, bytes);
    if (!buffer) return nullptr;
    owner->version.allocation = ++next_id;
    buffer->buft = type;
    buffer->context = owner.release();
    buffer->iface.free_buffer = free_watched;
    buffer->iface.get_base = [](ggml_backend_buffer_t b) { return static_cast<WatchedBuffer *>(b->context)->data; };
    buffer->iface.clear = [](ggml_backend_buffer_t b, uint8_t value) {
        std::memset(static_cast<WatchedBuffer *>(b->context)->data, value, b->size);
    };
    return buffer;
}

WeightVersion weight_version(ggml_backend_buffer_t buffer, const uint8_t * raw, size_t bytes) {
    if (!buffer || buffer->iface.free_buffer != free_watched ||
        buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return {};
    auto & state = *static_cast<WatchedBuffer *>(buffer->context);
    std::lock_guard<std::mutex> guard(state.mutex);
    // Preserve the generation of each page across all streams and overlapping
    // tensor views. A split projection only has to query the rows on the NPU.
    if (state.pages.empty()) {
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        state.page_size = system.dwPageSize;
        const size_t pages = 1 + (buffer->size - 1) / state.page_size;
        state.page_changes.resize(pages);
        state.pages.resize(pages);
    }
    const auto base = reinterpret_cast<uintptr_t>(state.data);
    const size_t offset = reinterpret_cast<uintptr_t>(raw) - base;
    if (!bytes || offset >= buffer->size || bytes > buffer->size - offset) return {};
    const size_t first = offset / state.page_size;
    const size_t end = 1 + (offset + bytes - 1) / state.page_size;
    ULONG_PTR count = end - first;
    void * start = reinterpret_cast<void *>(base + first * state.page_size);
    DWORD granularity{};
    if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, start, count * state.page_size, state.pages.data(), &count, &granularity)) {
        std::fill(state.page_changes.begin() + first, state.page_changes.begin() + end, ++state.version.changes);
        return {};
    }
    if (count) {
        const auto changed = ++state.version.changes;
        for (size_t i = 0; i < count; ++i) {
            state.page_changes[(reinterpret_cast<uintptr_t>(state.pages[i]) - base) / state.page_size] = changed;
        }
    }
    return {state.version.allocation,
            *std::max_element(state.page_changes.begin() + first, state.page_changes.begin() + end)};
}

ggml_backend_buffer_type_t host_weights(ggml_backend_dev_t dev) {
    // Preserve standard row layout while allowing llama's loader to reserve a
    // language-model head before it considers the CPU-specific repacked types.
    static ggml_backend_buffer_type type = [dev] {
        auto host = *ggml_backend_cpu_buffer_type();
        host.iface.get_name = [](ggml_backend_buffer_type_t) { return "Phoenix_GGUF"; };
        host.iface.alloc_buffer = allocate_host_weights;
        host.device = dev;
        return host;
    }();
    return &type;
}

bool useful_shape(const ggml_tensor * weight) {
    const int64_t cols = weight->ne[0], rows = weight->ne[1];
    switch (placement()) {
        case Placement::head:
            if (!output_weight(weight) || cols < 128 || cols > 8192 || rows < 512 || rows > 262144) return false;
            return PhoenixMatvec::packed_bytes(rows, cols) <= size_t(256) * 1024 * 1024;
        case Placement::split_head:
            return output_weight(weight) && rows >= 512 && rows <= 262144 && cols >= 128 && cols <= 8192;
        case Placement::dense:
            return cols >= 768 && cols <= 1024 && rows >= 2048 && rows <= 3072;
        case Placement::all:
            return cols >= 128 && cols <= 8192 && rows >= 128 && rows <= 24576 &&
                   ((cols + 1023) / 1024) * ((rows + 3071) / 3072) <= 8;
    }
    return false;
}

bool supported(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (metadata_op(op->op)) return true;
    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * vector = op->src[1];
    const bool empty_weight_probe = head_mode() && weight && weight->buffer && !weight->data &&
        ggml_backend_buffer_get_type(weight->buffer) == host_weights(dev) &&
        ggml_backend_buffer_get_size(weight->buffer) == 0;
    // Tied vocabulary embeddings must share the same loader context as the
    // head. Real embedding lookups are performed by CPU on this host buffer.
    if (op->op == GGML_OP_GET_ROWS) {
        return empty_weight_probe && vector && vector->type == GGML_TYPE_I32 && vector->ne[0] == 512 &&
               !std::strcmp(weight->name, "token_embd.weight") && useful_shape(weight);
    }
    if (op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32 ||
        op->op_params[0] != GGML_PREC_UNDEFINED || op->op_params[1] == GGML_HINT_SRC0_IS_HADAMARD) return false;

    if (!weight || !vector || vector->type != GGML_TYPE_F32 || weight->op != GGML_OP_NONE ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(vector) ||
        !ggml_is_contiguous(op)) return false;
    const bool loader_probe = empty_weight_probe && vector->ne[1] == 512;
    const int64_t cols = weight->ne[0], rows = weight->ne[1];
    if (ggml_is_empty(op) || vector->ne[1] < 1 || weight->ne[2] != 1 || weight->ne[3] != 1 || vector->ne[0] != cols ||
        vector->ne[2] != 1 || vector->ne[3] != 1 ||
        op->ne[0] != rows || op->ne[1] != vector->ne[1] || op->ne[2] != 1 || op->ne[3] != 1) return false;

    // Bionic's pinned AVX2 CPU keeps Q5_0 in canonical GGUF layout. Reserve
    // weights of the tested prompt shape in our write-watched host allocation,
    // while preserving native CPU repacking for every other tensor format.
    const bool small_ffn = rows == 2560 && cols == 960;
    const bool prompt_weights = loader_probe && small_ffn && weight->type == GGML_TYPE_Q5_0 && prefill_enabled() &&
                                PhoenixPrefill::supports(rows, cols, 128);
    const bool prompt = !loader_probe && vector->ne[1] > 1;
    const bool large_ffn = prompt && !small_ffn && large_prompt_weight(*weight);
    if (prompt) {
        if (!prefill_enabled() || (!small_ffn && !large_ffn) ||
            !PhoenixPrefill::supports(rows, cols, vector->ne[1])) return false;
    } else if (!prompt_weights && (!useful_shape(weight) ||
               (placement() == Placement::split_head && !ggml_get_type_traits_cpu(weight->type)->vec_dot))) {
        return false;
    }
    if (weight->extra || (weight->buffer && !ggml_backend_buffer_is_host(weight->buffer))) {
        // In the pinned AVX2 runtime these shapes always select Q4_0 8x8.
        if (!(large_ffn && weight->extra && weight->buffer &&
              ggml_backend_buffer_get_type(weight->buffer) == cpu_repack_type())) return false;
    }

    const auto * traits = ggml_get_type_traits(weight->type);
    return weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_BF16 ||
           ((weight->type == GGML_TYPE_F16 || traits->is_quantized) && traits->to_float);
}

struct WeightRows {
    const ggml_tensor & weight;

    static void decode(void * source, size_t index, uint16_t * output) {
        auto & self = *static_cast<WeightRows *>(source);
        const auto & weight = self.weight;
        const auto * bytes = static_cast<const uint8_t *>(weight.data) + index * weight.nb[1];
        const size_t cols = static_cast<size_t>(weight.ne[0]);
        if (weight.type == GGML_TYPE_Q5_0) {
            PhoenixPrefill::decode_q5_0_row(bytes, output, cols);
            return;
        }
        if (weight.type == GGML_TYPE_BF16) {
            std::memcpy(output, bytes, cols * sizeof(uint16_t));
            return;
        }
        const float * floats = reinterpret_cast<const float *>(bytes);
        if (weight.type != GGML_TYPE_F32) {
            thread_local std::vector<float> row;
            row.resize(cols);
            ggml_get_type_traits(weight.type)->to_float(bytes, row.data(), cols);
            floats = row.data();
        }
        ggml_fp32_to_bf16_row(floats, reinterpret_cast<ggml_bf16_t *>(output), cols);
    }
};

template <class Engine> struct CachedWeight {
    size_t rows;
    size_t cols;
    ggml_type type;
    bool repacked;
    size_t cost;
    WeightVersion version;
    std::vector<uint8_t> source;
    std::shared_ptr<const typename Engine::Prepared> prepared;
};

template <class Engine> using WeightCache = std::unordered_map<const void *, CachedWeight<Engine>>;

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Run the untouched rows of a vocabulary projection using the bundled CPU
// implementation. The graph, quantization scratch and sleeping worker threads
// are reused across tokens. The two devices write disjoint rows of the logits.
class CpuRemainder {
public:
    CpuRemainder(ggml_type type, size_t rows, size_t cols, int threads) : threads_(threads) {
        constexpr size_t graph_size = 8;
        const size_t overhead = 3 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);
        ctx_.reset(ggml_init({overhead, nullptr, true}));
        if (!ctx_) throw std::runtime_error("CPU graph allocation failed");
        weight_ = ggml_new_tensor_2d(ctx_.get(), type, cols, rows);
        vector_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, cols);
        output_ = ggml_mul_mat(ctx_.get(), weight_, vector_);
        graph_ = ggml_new_graph_custom(ctx_.get(), graph_size, false);
        ggml_build_forward_expand(graph_, output_);
        auto params = ggml_threadpool_params_default(threads);
        params.poll = 0; // This pool is idle during the rest of the model.
        pool_.reset(ggml_threadpool_new(&params));
        if (!pool_) throw std::runtime_error("CPU thread pool allocation failed");
        plan_ = ggml_graph_plan(graph_, threads, pool_.get());
        scratch_.resize(plan_.work_size);
        plan_.work_data = scratch_.data();
        worker_ = std::thread([this] { run(); });
    }

    ~CpuRemainder() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_one();
        worker_.join();
    }

    bool matches(ggml_type type, size_t rows, size_t cols, int threads) const {
        return weight_->type == type && weight_->ne[0] == cols && weight_->ne[1] == rows && threads_ == threads;
    }

    void start(void * weights, const float * vector, float * output) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            weight_->data = weights;
            vector_->data = const_cast<float *>(vector);
            output_->data = output;
            pending_ = true;
        }
        wake_.notify_one();
    }

    ggml_status finish() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return !pending_; });
        return status_;
    }

    double compute_ms() const { return compute_ms_; }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            wake_.wait(lock, [this] { return pending_ || stop_; });
            if (stop_) return;
            lock.unlock();
            const auto start = Clock::now();
            const auto status = ggml_graph_compute(graph_, &plan_);
            lock.lock();
            compute_ms_ = elapsed_ms(start);
            status_ = status;
            pending_ = false;
            done_.notify_one();
        }
    }

    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_{nullptr, ggml_free};
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool_{nullptr, ggml_threadpool_free};
    ggml_tensor * weight_ = nullptr;
    ggml_tensor * vector_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_cplan plan_{};
    std::vector<uint8_t> scratch_;
    int threads_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    bool stop_ = false, pending_ = false;
    ggml_status status_ = GGML_STATUS_SUCCESS;
    double compute_ms_ = 0;
};

size_t device_rows(size_t rows) {
    if (placement() != Placement::split_head) return rows;
    static const int tiles = [] {
        const char * value = std::getenv("PHOENIX_XDNA_HEAD_TILES");
        const int n = value ? std::atoi(value) : 1;
        return n >= 1 && n <= 32 ? n : 1;
    }();
    return (std::min)(rows, size_t(tiles) * 3072);
}

struct Stream {
    std::vector<uint8_t> activation_codes;
    std::vector<float> activation_values;

    // Stock quantized CPU dot products round the incoming vector once. Apply
    // the same rounding to the NPU's rows of a shared vocabulary projection.
    const float * dot_vector(ggml_type weight_type, const float * input, size_t cols) {
        const auto type = ggml_get_type_traits_cpu(weight_type)->vec_dot_type;
        if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q8_K) return input;
        activation_codes.resize(ggml_row_size(type, cols));
        activation_values.resize(cols);
        ggml_get_type_traits_cpu(type)->from_float(input, activation_codes.data(), cols);
        if (type == GGML_TYPE_Q8_K) {
            dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(activation_codes.data()), activation_values.data(), cols);
        } else {
            ggml_get_type_traits(type)->to_float(activation_codes.data(), activation_values.data(), cols);
        }
        return activation_values.data();
    }
    std::unique_ptr<PhoenixMatvec> npu;
    std::unique_ptr<PhoenixPrefill> prefill;
    std::unique_ptr<CpuRemainder> cpu;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> fallback_cpu{nullptr, ggml_backend_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> fallback_context{nullptr, ggml_free};
    ggml_cgraph * fallback_graph = nullptr;
    int cpu_threads = GGML_DEFAULT_N_THREADS;
    WeightCache<PhoenixMatvec> cache;
    WeightCache<PhoenixPrefill> prompt_cache;
    size_t cached_bytes = 0;
    size_t compact_bytes = 0;
    unsigned long long hits = 0;
    unsigned long long uploads = 0;
    unsigned long long dispatches = 0;
    unsigned long long prompt_dispatches = 0;
    std::vector<std::pair<size_t, size_t>> traced_prompt_shapes;
    double validation_ms = 0;
    double preparation_ms = 0;
    double compute_ms = 0;
    double cpu_ms = 0;
    double wall_ms = 0;

    template <class Engine>
    std::shared_ptr<const typename Engine::Prepared> get_weight(
        const ggml_tensor & weight, size_t rows, Engine & engine, WeightCache<Engine> & entries) {
        const auto start = Clock::now();
        const size_t cols = static_cast<size_t>(weight.ne[0]);
        const size_t bytes = rows * weight.nb[1];
        const bool compact = std::is_same_v<Engine, PhoenixPrefill> && weight.type == GGML_TYPE_Q4_0 && cols != 960;
        const auto * raw = static_cast<const uint8_t *>(weight.data);
        const auto version = weight_version(weight.buffer, raw, bytes);
        auto found = entries.find(raw);
        if (found != entries.end()) {
            const auto & entry = found->second;
            const bool same_version = version.allocation && version.allocation == entry.version.allocation &&
                version.changes == entry.version.changes;
            if (entry.rows == rows && entry.cols == cols && entry.type == weight.type &&
                entry.repacked == (weight.extra != nullptr) &&
                (same_version || (!version.allocation && entry.source.size() == bytes &&
                                  std::memcmp(raw, entry.source.data(), bytes) == 0))) {
                ++hits;
                validation_ms += elapsed_ms(start);
                return entry.prepared;
            }
            cached_bytes -= entry.cost;
            if constexpr (std::is_same_v<Engine, PhoenixPrefill>) {
                if (entry.type == GGML_TYPE_Q4_0 && entry.cols != 960) compact_bytes -= entry.cost;
            }
            entries.erase(found);
        }
        validation_ms += elapsed_ms(start);
        const auto prep_start = Clock::now();
        const size_t cost = (version.allocation ? 0 : bytes) + Engine::packed_bytes(rows, cols);
        constexpr size_t headroom = size_t(68) << 20;
        size_t resident_limit = (std::max)(size_t(768) << 20, compact_bytes + headroom);
        if (compact) {
            auto budget = large_prompt_budget(rows, cols);
            if (budget > prompt_budget()) {
                // Recheck before growth. Other model processes may have
                // consumed RAM after this stream selected its initial profile.
                if (available_memory(uint64_t(4) << 30)) resident_limit = budget + headroom;
                else budget = prompt_budget();
            }
            if (cost > budget || compact_bytes > budget - cost ||
                cost > resident_limit || cached_bytes > resident_limit - cost) return {};
        }
        WeightRows input{weight};
        std::shared_ptr<const typename Engine::Prepared> prepared;
        if constexpr (std::is_same_v<Engine, PhoenixPrefill>) {
            if (compact) prepared = engine.prepare_q4(rows, cols, raw, weight.extra != nullptr);
        }
        if (!prepared) prepared = engine.prepare(rows, cols, WeightRows::decode, &input);
        if (cost <= resident_limit && cached_bytes <= resident_limit - cost) {
            std::vector<uint8_t> snapshot;
            if (!version.allocation) snapshot.assign(raw, raw + bytes);
            entries.emplace(raw, CachedWeight<Engine>{rows, cols, weight.type, weight.extra != nullptr, cost, version,
                            std::move(snapshot), prepared});
            cached_bytes += cost;
            if (compact) compact_bytes += cost;
        }
        ++uploads;
        preparation_ms += elapsed_ms(prep_start);
        return prepared;
    }

    void multiply_prompt(ggml_tensor * op, bool adjacent_shared_input) {
        if (!prefill) prefill = std::make_unique<PhoenixPrefill>();
        prefill->set_prepare_threads(cpu_threads);
        const auto & weight = *op->src[0];
        const size_t rows = weight.ne[1], cols = weight.ne[0], tokens = op->src[1]->ne[1];
        auto packed = get_weight(weight, rows, *prefill, prompt_cache);
        if (!packed) {
            // A GGUF may mix eligible layer sizes differently from the profiled
            // models. Use its original CPU_REPACK tensor if it exhausts the cap.
            if (!fallback_cpu) {
                decltype(fallback_cpu) backend(ggml_backend_cpu_init(), ggml_backend_free);
                decltype(fallback_context) context(ggml_init({ggml_graph_overhead_custom(1, false), nullptr, true}), ggml_free);
                if (!backend || !context) throw std::runtime_error("CPU fallback allocation failed");
                auto * graph = ggml_new_graph_custom(context.get(), 1, false);
                if (!graph) throw std::runtime_error("CPU fallback graph allocation failed");
                fallback_cpu = std::move(backend);
                fallback_context = std::move(context);
                fallback_graph = graph;
            }
            ggml_backend_cpu_set_n_threads(fallback_cpu.get(), cpu_threads);
            ggml_graph_clear(fallback_graph);
            ggml_graph_add_node(fallback_graph, op);
            if (ggml_backend_graph_compute(fallback_cpu.get(), fallback_graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("CPU prefill fallback failed");
            return;
        }
        const auto start = Clock::now();
        const auto * input = static_cast<const float *>(op->src[1]->data);
        const bool reused = prefill->compute(*packed, input, static_cast<float *>(op->data), tokens, weight.type);
        const double ms = elapsed_ms(start);
        ++dispatches;
        ++prompt_dispatches;
        const auto shape = std::make_pair(rows, cols);
        const bool first_shape = std::find(traced_prompt_shapes.begin(), traced_prompt_shapes.end(), shape) == traced_prompt_shapes.end();
        if (first_shape) traced_prompt_shapes.push_back(shape);
        if (first_shape || (prompt_dispatches & (prompt_dispatches - 1)) == 0) {
            char line[512];
            std::snprintf(line, sizeof(line), "[Phoenix XDNA1] prefill pid=%lu count=%llu tensor=%.*s rows=%zu cols=%zu tokens=%zu type=%s buffer=%s adjacent_shared_input=%d input_reused=%d cached_mib=%zu hits=%llu uploads=%llu validation_ms=%.3f compute_ms=%.3f\n",
                          GetCurrentProcessId(), prompt_dispatches, GGML_MAX_NAME, weight.name, rows, cols, tokens,
                          ggml_type_name(weight.type), weight.buffer ? ggml_backend_buffer_name(weight.buffer) : "none",
                          int(adjacent_shared_input), int(reused), cached_bytes >> 20, hits, uploads, validation_ms, ms);
            trace(line);
        }
    }

    void multiply(ggml_tensor * op) {
        const ggml_tensor * weight = op->src[0];
        const size_t cols = static_cast<size_t>(weight->ne[0]);
        const size_t rows = static_cast<size_t>(weight->ne[1]);
        if (!npu) npu = std::make_unique<PhoenixMatvec>();
        const size_t npu_rows = device_rows(rows);
        const size_t cpu_rows = rows - npu_rows;
        const auto * vector = static_cast<const float *>(op->src[1]->data);
        auto * output = static_cast<float *>(op->data);
        const auto begin = Clock::now();
        if (cpu_rows) {
            if (!cpu || !cpu->matches(weight->type, cpu_rows, cols, cpu_threads)) {
                cpu = std::make_unique<CpuRemainder>(weight->type, cpu_rows, cols, cpu_threads);
            }
            cpu->start(static_cast<uint8_t *>(weight->data) + weight->nb[1] * npu_rows, vector, output + npu_rows);
        }
        try {
            auto packed = get_weight(*weight, npu_rows, *npu, cache);
            const auto start = Clock::now();
            npu->compute(*packed, dot_vector(weight->type, vector, cols), output);
            compute_ms += elapsed_ms(start);
        } catch (...) {
            if (cpu_rows) cpu->finish();
            throw;
        }
        if (cpu_rows) {
            if (cpu->finish() != GGML_STATUS_SUCCESS) throw std::runtime_error("CPU vocabulary projection failed");
            cpu_ms += cpu->compute_ms();
        }
        wall_ms += elapsed_ms(begin);
        ++dispatches;
        const auto count = ++completed;
        if ((count & (count - 1)) == 0) {
            MEMORY_BASIC_INFORMATION source{};
            VirtualQuery(weight->data, &source, sizeof(source));
            const char * memory = source.Type == MEM_MAPPED ? "mapped" : source.Type == MEM_PRIVATE ? "private" : "other";
            char line[512];
            std::snprintf(line, sizeof(line),
                          "[Phoenix XDNA1] dispatch pid=%lu count=%llu tensor=%.*s rows=%zu cpu_rows=%zu cols=%zu type=%s buffer=%s memory=%s protect=0x%lx hits=%llu uploads=%llu cached_mib=%zu validation_ms=%.3f prepare_ms=%.3f compute_ms=%.3f cpu_ms=%.3f wall_ms=%.3f\n",
                          GetCurrentProcessId(), count, GGML_MAX_NAME, weight->name, npu_rows, cpu_rows, cols,
                          ggml_type_name(weight->type), weight->buffer ? ggml_backend_buffer_name(weight->buffer) : "none",
                          memory, source.Protect, hits, uploads, cached_bytes >> 20,
                          validation_ms, preparation_ms, compute_ms, cpu_ms, wall_ms);
            trace(line);
        }
    }
};

ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto & state = *static_cast<Stream *>(backend->context);
    try {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * op = ggml_graph_node(graph, i);
            if (!(op->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_is_empty(op) || metadata_op(op->op)) continue;
            if (!supported(backend->device, op)) throw std::runtime_error("unsupported scheduled operation");
            if (op->src[1]->ne[1] > 1) {
                const auto * next = i + 1 < ggml_graph_n_nodes(graph) ? ggml_graph_node(graph, i + 1) : nullptr;
                const bool shares_input = next && (next->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    next->op == GGML_OP_MUL_MAT && next->src[1] == op->src[1] && next->src[0] != op->src[0] &&
                    ggml_are_same_shape(next->src[0], op->src[0]);
                state.multiply_prompt(op, shares_input);
            } else {
                state.multiply(op);
            }
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::exception & e) {
        char line[768];
        std::snprintf(line, sizeof(line), "[Phoenix XDNA1] error pid=%lu reason=%s\n",
                      GetCurrentProcessId(), e.what());
        trace(line);
        return GGML_STATUS_FAILED;
    }
}

ggml_backend_t init_stream(ggml_backend_dev_t device, const char *) {
    ggml_backend_i api{};
    api.get_name = [](ggml_backend_t) { return static_cast<const char *>(NAME); };
    api.free = [](ggml_backend_t b) {
        auto * state = static_cast<Stream *>(b->context);
        if (state->dispatches) {
            char line[128];
            std::snprintf(line, sizeof(line), "[Phoenix XDNA1] closed pid=%lu dispatches=%llu\n",
                          GetCurrentProcessId(), state->dispatches);
            trace(line);
        } else {
            trace("[Phoenix XDNA1] stream closed with zero completed NPU operations\n");
        }
        delete state;
        delete b;
    };
    api.graph_compute = compute;
    std::unique_ptr<Stream> state(new (std::nothrow) Stream{});
    if (!state) return nullptr;
    auto * backend = new (std::nothrow) ggml_backend{&guid, api, device, state.get()};
    if (backend) state.release();
    return backend;
}

ggml_backend_dev_t get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) return nullptr;
    static ggml_backend_device device = [reg] {
        ggml_backend_device_i api{};
        api.get_name = [](ggml_backend_dev_t) { return static_cast<const char *>(NAME); };
        api.get_description = [](ggml_backend_dev_t) { return static_cast<const char *>(DESCRIPTION); };
        api.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; };
        api.get_memory = [](ggml_backend_dev_t, size_t * available, size_t * total) {
            *available = *total = 0; // XDNA 1 uses system memory.
        };
        api.get_props = [](ggml_backend_dev_t, ggml_backend_dev_props * props) {
            *props = {};
            props->name = NAME;
            props->description = DESCRIPTION;
            props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
            props->caps.mmap_support = true;
            // Only head-mode's named host buffer is loaded into owned,
            // write-watched pages. All other GGUF buffers retain their loader.
            props->caps.buffer_from_host_ptr = !head_mode();
        };
        api.init_backend = init_stream;
        api.get_buffer_type = [](ggml_backend_dev_t dev) {
            return head_mode() ? host_weights(dev) : ggml_backend_cpu_buffer_type();
        };
        api.buffer_from_host_ptr = [](ggml_backend_dev_t, void * ptr, size_t bytes, size_t) {
            return ggml_backend_cpu_buffer_from_ptr(ptr, bytes);
        };
        api.supports_op = supported;
        api.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t type) {
            return ggml_backend_buft_is_host(type) || (prefill_enabled() && type == cpu_repack_type());
        };
        return ggml_backend_device{api, reg, nullptr};
    }();
    return &device;
}

ggml_backend_reg_t registration() {
    static ggml_backend_reg reg{
        GGML_BACKEND_API_VERSION,
        {[](ggml_backend_reg_t) { return static_cast<const char *>(NAME); },
         [](ggml_backend_reg_t) -> size_t { return 1; }, get_device,
         [](ggml_backend_reg_t, const char * name) -> void * {
             if (std::strcmp(name, "ggml_backend_set_n_threads") != 0) return nullptr;
             ggml_backend_set_n_threads_t setter = [](ggml_backend_t backend, int threads) {
                 static_cast<Stream *>(backend->context)->cpu_threads =
                     (std::max)(1, (std::min)(threads, GGML_MAX_N_THREADS));
             };
             return reinterpret_cast<void *>(setter);
         }},
        nullptr,
    };
    return &reg;
}

}  // namespace

GGML_BACKEND_DL_IMPL(registration)
