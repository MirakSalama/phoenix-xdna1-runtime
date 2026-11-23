// Exercise GGUF-style quantized rows through the real ggml scheduler.
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void require(bool good, const char * why) {
    if (!good) throw std::runtime_error(why);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3 || (argc == 4 && !std::strcmp(argv[3], "mapped")),
                "expected stock CPU directory, Phoenix DLL path, and optional 'mapped'");
        const bool mapped = argc == 4;
        ggml_backend_load_all_from_path(argv[1]);
        require(ggml_backend_load(argv[2]), "Phoenix DLL failed to load");
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_t npu = ggml_backend_init_by_name("XDNA1", nullptr);
        require(cpu && npu, "backend registration failed");

        const char * mode = std::getenv("PHOENIX_XDNA_PLACEMENT");
        const bool head = !mode || (std::strcmp(mode, "all") && std::strcmp(mode, "dense"));
        const bool broad = mode && std::strcmp(mode, "all") == 0;
        const int rows = head ? 8200 : broad ? 257 : 2053;
        const int cols = head ? 256 : broad ? 1280 : 768;
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
        auto * ctx = ggml_init({ggml_graph_overhead() + 8 * ggml_tensor_overhead(), nullptr, true});
        auto * probe_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 512);
        auto * probe = ggml_mul_mat(ctx, weight, probe_input);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
        auto * embedding = ggml_get_rows(ctx, weight, ids);
        ggml_set_name(weight, "blk.0.ffn_gate.weight");
        if (head) {
            auto * buft = ggml_backend_get_default_buffer_type(npu);
            require(!std::strcmp(ggml_backend_buft_name(buft), "Phoenix_GGUF") &&
                    ggml_backend_buft_is_host(buft), "expected standard-layout Phoenix weight buffer");
            weight->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
            require(!ggml_backend_supports_op(npu, probe), "internal FFN reserved as head");
            for (const char * name : {"token_embd.weight", "output.weight"}) {
                ggml_set_name(weight, name);
                require(ggml_backend_supports_op(npu, probe), "head not reserved before GGUF repacking");
                require(ggml_backend_supports_op(npu, embedding) == !std::strcmp(name, "token_embd.weight"),
                        "tied embedding must share the head's buffer context");
            }
            ggml_backend_buffer_free(weight->buffer);
            weight->buffer = nullptr;
        }

        ggml_backend_buffer_t backing = nullptr;
        ggml_backend_buffer_t weight_buffer = nullptr;
        if (mapped) {
            backing = ggml_backend_alloc_buffer(cpu, codes.size());
            require(backing, "mapped backing allocation failed");
        }
        auto allocate_weights = [&] {
            if (backing) {
                void * data = ggml_backend_buffer_get_base(backing);
                weight_buffer = ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(npu), data, codes.size(), codes.size());
                require(weight_buffer && ggml_backend_tensor_alloc(weight_buffer, weight, data) == GGML_STATUS_SUCCESS,
                        "mapped weight tensor allocation failed");
            } else {
                weight_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, head ? npu : cpu);
                require(weight_buffer, "weight allocation failed");
            }
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        };
        allocate_weights();
        auto empty_vectors = *probe_input;
        empty_vectors.ne[1] = 0;
        auto empty_output = *probe;
        empty_output.ne[1] = 0;
        empty_output.src[1] = &empty_vectors;
        require(!ggml_backend_supports_op(npu, &empty_output), "empty batch incorrectly scheduled on the NPU");
        require(!ggml_backend_supports_op(npu, probe), "real 512-token batch incorrectly accepted");
        require(!ggml_backend_supports_op(npu, embedding), "real embedding lookup incorrectly accepted");
        auto * graph = ggml_new_graph(ctx);
        auto * single = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cols);
        auto * batch = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, 2);
        ggml_set_input(single);
        ggml_set_input(batch);
        ggml_set_input(ids);
        auto * decode = ggml_mul_mat(ctx, weight, single);
        auto * prefill = ggml_mul_mat(ctx, weight, batch);
        ggml_set_output(decode);
        ggml_set_output(prefill);
        ggml_set_output(embedding);
        ggml_build_forward_expand(graph, decode);
        ggml_build_forward_expand(graph, prefill);
        ggml_build_forward_expand(graph, embedding);
        ggml_backend_t devices[] = {npu, cpu};
        auto scheduler = ggml_backend_sched_new(devices, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        require(ggml_backend_sched_alloc_graph(scheduler, graph), "scheduler allocation failed");
        require(ggml_backend_sched_get_tensor_backend(scheduler, decode) == npu, "decode not on NPU");
        require(ggml_backend_sched_get_tensor_backend(scheduler, prefill) == cpu, "batch not on CPU");
        require(ggml_backend_sched_get_tensor_backend(scheduler, embedding) == cpu, "embedding lookup not on CPU");
        for (int pass = 0; pass < 5; ++pass) {
            // Reuse, change with GGML, change via host pointer, then replace a
            // weight buffer while keeping the same tensor and execution stream.
            if (pass == 4) {
                const auto * old_data = weight->data;
                ggml_backend_buffer_free(weight_buffer);
                weight->buffer = nullptr;
                weight->data = nullptr;
                allocate_weights();
                std::printf("replacement_weight_same_address=%d\n", old_data == weight->data);
                require(!mapped || old_data == weight->data, "borrowed address was not reused");
            }
            if (pass != 1) {
                if (pass >= 2) for (float & value : source) value *= -0.5f;
                require(ggml_quantize_chunk(type, source.data(), codes.data(), 0, rows, cols, nullptr) ==
                        codes.size(), "Q4_K quantization failed");
                if (pass == 3) std::memcpy(weight->data, codes.data(), codes.size());
                else ggml_backend_tensor_set(weight, codes.data(), 0, codes.size());
            } else {
                for (int i = 0; i < cols; ++i) values[cols + i] = values[i] = sample(generator);
            }
            // The graph allocator may reuse each input's storage after its last consumer.
            // Normal llama.cpp decode submits fresh input data for every graph execution.
            ggml_backend_tensor_set(single, values.data(), 0, cols * sizeof(float));
            ggml_backend_tensor_set(batch, values.data(), 0, 2 * cols * sizeof(float));
            const std::vector<int32_t> indices(512, rows - 1);
            ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(int32_t));
            require(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS, "graph failed");
            std::vector<float> on_npu(rows), on_cpu(rows * 2);
            ggml_backend_tensor_get(decode, on_npu.data(), 0, rows * sizeof(float));
            ggml_backend_tensor_get(prefill, on_cpu.data(), 0, rows * 2 * sizeof(float));
            std::vector<float> embedded(cols);
            ggml_backend_tensor_get(embedding, embedded.data(), 0, cols * sizeof(float));
            ggml_get_type_traits(type)->to_float(codes.data() + (rows - 1) * stride, dequant.data(), cols);
            for (int i = 0; i < cols; ++i) require(std::abs(embedded[i] - dequant[i]) < 0.00001f, "GGUF embedding mismatch");
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
        if (backing) ggml_backend_buffer_free(backing);
        ggml_free(weights_ctx);
        ggml_backend_free(npu);
        ggml_backend_free(cpu);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
