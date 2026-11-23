// Phoenix XDNA 1 is a synchronous ggml accelerator alongside the CPU backend.
// ggml owns graph placement, GGUF loading, tokenization and unsupported ops.
#include "ggml-backend-impl.h"
#include "phoenix_matvec.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
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

bool supported(ggml_backend_dev_t, const ggml_tensor * op) {
    if (metadata_op(op->op)) return true;
    if (op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32 ||
        op->op_params[0] != GGML_PREC_UNDEFINED || op->op_params[1] == GGML_HINT_SRC0_IS_HADAMARD) return false;

    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * vector = op->src[1];
    if (!weight || !vector || vector->type != GGML_TYPE_F32 || weight->op != GGML_OP_NONE ||
        weight->extra || !ggml_is_contiguous(weight) || !ggml_is_contiguous(vector) ||
        !ggml_is_contiguous(op)) return false;
    if (weight->buffer && ggml_backend_buffer_get_type(weight->buffer) != ggml_backend_cpu_buffer_type()) {
        return false;  // A backend-specific repacked buffer is not a GGUF row.
    }
    const int64_t cols = weight->ne[0], rows = weight->ne[1];
    if (weight->ne[2] != 1 || weight->ne[3] != 1 || vector->ne[0] != cols ||
        vector->ne[1] != 1 || vector->ne[2] != 1 || vector->ne[3] != 1 ||
        op->ne[0] != rows || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1) return false;

    // Each hardware job stages a fixed 3072 x 1024 tile. Keep very small
    // projections and large heads on the CPU, and bound conversion workspace.
    if (cols < 128 || cols > 8192 || rows < 128 || rows > 24576 ||
        ((cols + 1023) / 1024) * ((rows + 3071) / 3072) > 8) return false;

    const auto * traits = ggml_get_type_traits(weight->type);
    return weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_BF16 ||
           ((weight->type == GGML_TYPE_F16 || traits->is_quantized) && traits->to_float);
}

struct Stream {
    std::unique_ptr<PhoenixMatvec> npu;
    std::vector<ggml_bf16_t> converted;
    std::vector<float> row;
    unsigned long long dispatches = 0;

    void multiply(ggml_tensor * op) {
        const ggml_tensor * weight = op->src[0];
        const size_t cols = static_cast<size_t>(weight->ne[0]);
        const size_t rows = static_cast<size_t>(weight->ne[1]);
        if (!npu) npu = std::make_unique<PhoenixMatvec>();
        const void * matrix = weight->data;
        if (weight->type != GGML_TYPE_BF16) {
            converted.resize(rows * cols);
            const auto * bytes = static_cast<const unsigned char *>(weight->data);
            if (weight->type == GGML_TYPE_F32) {
                ggml_fp32_to_bf16_row(static_cast<const float *>(weight->data),
                                      converted.data(), rows * cols);
            } else {
                const auto decode = ggml_get_type_traits(weight->type)->to_float;
                row.resize(cols);
                for (size_t i = 0; i < rows; ++i) {
                    decode(bytes + i * weight->nb[1], row.data(), cols);
                    ggml_fp32_to_bf16_row(row.data(), converted.data() + i * cols, cols);
                }
            }
            matrix = converted.data();
        }
        npu->compute(static_cast<const uint16_t *>(matrix),
                     static_cast<const float *>(op->src[1]->data),
                     static_cast<float *>(op->data), rows, cols);
        ++dispatches;
        const auto count = ++completed;
        if ((count & (count - 1)) == 0) {
            char line[512];
            std::snprintf(line, sizeof(line),
                          "[Phoenix XDNA1] dispatch pid=%lu count=%llu tensor=%.*s rows=%zu cols=%zu type=%s\n",
                          GetCurrentProcessId(), count, GGML_MAX_NAME, weight->name, rows, cols,
                          ggml_type_name(weight->type));
            trace(line);
        }
    }
};

ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto & state = *static_cast<Stream *>(backend->context);
    try {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * op = ggml_graph_node(graph, i);
            if (!(op->flags & GGML_TENSOR_FLAG_COMPUTE) || metadata_op(op->op)) continue;
            if (!supported(backend->device, op)) throw std::runtime_error("unsupported scheduled operation");
            state.multiply(op);
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
            props->caps.buffer_from_host_ptr = true;
        };
        api.init_backend = init_stream;
        api.get_buffer_type = [](ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); };
        api.buffer_from_host_ptr = [](ggml_backend_dev_t, void * ptr, size_t bytes, size_t) {
            return ggml_backend_cpu_buffer_from_ptr(ptr, bytes);
        };
        api.supports_op = supported;
        api.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t type) {
            return type == ggml_backend_cpu_buffer_type();
        };
        return ggml_backend_device{api, reg, nullptr};
    }();
    return &device;
}

ggml_backend_reg_t registration() {
    static ggml_backend_reg reg{
        GGML_BACKEND_API_VERSION,
        {[](ggml_backend_reg_t) { return static_cast<const char *>(NAME); },
         [](ggml_backend_reg_t) -> size_t { return 1; }, get_device, nullptr},
        nullptr,
    };
    return &reg;
}

}  // namespace

GGML_BACKEND_DL_IMPL(registration)
