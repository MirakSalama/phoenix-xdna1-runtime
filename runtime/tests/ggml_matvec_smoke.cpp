// Exercise GGUF-style quantized rows through the real ggml scheduler.
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void require(bool good, const char * why) {
    if (!good) throw std::runtime_error(why);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "expected directory of stock CPU pack and path to Phoenix backend DLL");
        ggml_backend_load_all_from_path(argv[1]);
        require(ggml_backend_load(argv[2]), "Phoenix DLL failed to load");
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_t npu = ggml_backend_init_by_name("XDNA1", nullptr);
        require(cpu && npu, "backend registration failed");

        constexpr int rows = 257, cols = 1280; // two hardware jobs, one partial column tile
        constexpr auto type = GGML_TYPE_Q4_K;
        const size_t stride = ggml_row_size(type, cols);
        std::vector<float> source(rows * cols), values(cols * 2), dequant(cols);
        std::vector<unsigned char> codes(rows * stride);
        std::mt19937 generator(87);
        std::uniform_real_distribution<float> sample(-0.4f, 0.4f);
        for (float & value : source) value = sample(generator);
        for (int i = 0; i < cols; ++i) values[cols + i] = values[i] = sample(generator);

        auto * weights_ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
        auto * weight = ggml_new_tensor_2d(weights_ctx, type, cols, rows);
        ggml_set_name(weight, "test.immutable-or-updated.weight");
        auto weight_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, cpu);
        require(weight_buffer, "weight allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * ctx = ggml_init({ggml_graph_overhead() + 8 * ggml_tensor_overhead(), nullptr, true});
        auto * graph = ggml_new_graph(ctx);
        auto * single = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cols);
        auto * batch = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 2);
        ggml_set_input(single);
        ggml_set_input(batch);
        auto * decode = ggml_mul_mat(ctx, weight, single);
        auto * prefill = ggml_mul_mat(ctx, weight, batch);
        ggml_set_output(decode);
        ggml_set_output(prefill);
        ggml_build_forward_expand(graph, decode);
        ggml_build_forward_expand(graph, prefill);
        ggml_backend_t devices[] = {npu, cpu};
        auto scheduler = ggml_backend_sched_new(devices, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        require(ggml_backend_sched_alloc_graph(scheduler, graph), "scheduler allocation failed");
        require(ggml_backend_sched_get_tensor_backend(scheduler, decode) == npu, "decode not on NPU");
        require(ggml_backend_sched_get_tensor_backend(scheduler, prefill) == cpu, "batch not on CPU");
        for (int pass = 0; pass < 2; ++pass) {
            // Also verify a caller can update a ggml weight in place without stale decoded values.
            if (pass == 1) for (float & value : source) value *= -0.5f;
            require(ggml_quantize_chunk(type, source.data(), codes.data(), 0, rows, cols, nullptr) ==
                    codes.size(), "Q4_K quantization failed");
            ggml_backend_tensor_set(weight, codes.data(), 0, codes.size());
            // The graph allocator may reuse each input's storage after its last consumer.
            // Normal llama.cpp decode submits fresh input data for every graph execution.
            ggml_backend_tensor_set(single, values.data(), 0, cols * sizeof(float));
            ggml_backend_tensor_set(batch, values.data(), 0, 2 * cols * sizeof(float));
            require(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS, "graph failed");
            std::vector<float> on_npu(rows), on_cpu(rows * 2);
            ggml_backend_tensor_get(decode, on_npu.data(), 0, rows * sizeof(float));
            ggml_backend_tensor_get(prefill, on_cpu.data(), 0, rows * 2 * sizeof(float));
            double scale = 0, npu_error = 0, cpu_difference = 0;
            for (int r = 0; r < rows; ++r) {
                ggml_get_type_traits(type)->to_float(codes.data() + r * stride, dequant.data(), cols);
                double ref = 0;
                for (int c = 0; c < cols; ++c) ref += double(dequant[c]) * values[c];
                npu_error += std::pow(on_npu[r] - ref, 2);
                cpu_difference += std::pow(on_npu[r] - on_cpu[r], 2);
                scale += ref * ref;
            }
            const double against_reference = std::sqrt(npu_error / scale);
            const double against_cpu = std::sqrt(cpu_difference / scale);
            std::printf("pass=%d decode=XDNA1 prefill=CPU q4_k rows=%d cols=%d relative_rms_ref=%.6f relative_rms_cpu=%.6f\n",
                        pass + 1, rows, cols, against_reference, against_cpu);
            require(against_reference < 0.008 && against_cpu < 0.025, "numeric differential failed");
        }
        ggml_backend_sched_free(scheduler);
        ggml_free(ctx);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(weights_ctx);
        ggml_backend_free(npu);
        ggml_backend_free(cpu);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
