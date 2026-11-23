// Exercise actual GGUF up-projection rows through the stock CPU and Phoenix.
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

int main(int argc, char ** argv) {
    try {
        require(argc == 6, "expected CPU pack, Phoenix DLL, q5_0, GGUF or raw weight, and token count");
        require(!std::strcmp(argv[3], "q5_0"), "expected a Q5_0 up projection of width 960");
        constexpr int rows = 2560, cols = 960;
        constexpr auto type = GGML_TYPE_Q5_0;
        const int tokens = std::stoi(argv[5]);
        require(tokens >= 96 && tokens <= 512, "expected 96 to 512 prompt tokens");
        ggml_backend_load_all_from_path(argv[1]);
        require(ggml_backend_load(argv[2]), "Phoenix DLL load failed");
        auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        auto * npu = ggml_backend_init_by_name("XDNA1", nullptr);
        require(cpu && npu, "backend init failed");
        ggml_backend_cpu_set_n_threads(cpu, 6);
        auto * wc = ggml_init({ggml_tensor_overhead(), nullptr, true});
        auto * weight = ggml_new_tensor_2d(wc, type, cols, rows);
        auto * wb = ggml_backend_alloc_ctx_tensors(wc, cpu);
        require(wb, "GGUF weight allocation failed");
        ggml_set_name(weight, "blk.0.ffn_up.weight");
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::ifstream file(argv[4], std::ios::binary);
        char magic[4]{};
        file.read(magic, sizeof magic);
        file.seekg(0);
        std::vector<uint8_t> codes;
        if (!std::memcmp(magic, "GGUF", sizeof magic)) {
            auto * gguf = gguf_init_from_file(argv[4], {true, nullptr});
            require(gguf, "Could not read the GGUF header");
            const auto id = gguf_find_tensor(gguf, weight->name);
            require(id >= 0 && gguf_get_tensor_type(gguf, id) == type &&
                    gguf_get_tensor_size(gguf, id) == ggml_nbytes(weight), "Unexpected GGUF projection");
            file.seekg(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id));
            codes.resize(gguf_get_tensor_size(gguf, id));
            gguf_free(gguf);
            require(bool(file.read(reinterpret_cast<char *>(codes.data()), codes.size())), "Could not read the GGUF projection");
        } else {
            codes.assign(std::istreambuf_iterator<char>(file), {});
        }
        require(codes.size() == ggml_nbytes(weight), "expected complete GGUF weight");
        ggml_backend_tensor_set(weight, codes.data(), 0, codes.size());
        auto * ctx = ggml_init({ggml_graph_overhead() + 3 * ggml_tensor_overhead(), nullptr, true});
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, tokens);
        auto * y = ggml_mul_mat(ctx, weight, x);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        auto * data = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        require(data && ggml_backend_supports_op(npu, y), "NPU rejected projection");
        std::mt19937 rng(105);
        std::uniform_real_distribution<float> dist(-0.75f, 0.75f);
        std::vector<float> input(cols * tokens), expected(rows * tokens), measured(expected.size());
        for (int pass = 0; pass < 3; ++pass) {
            if (pass == 2) {
                // Exchange two full GGUF rows, including all block scales.
                std::swap_ranges(codes.begin(), codes.begin() + weight->nb[1], codes.end() - weight->nb[1]);
                ggml_backend_tensor_set(weight, codes.data(), 0, codes.size());
            }
            for (auto & value : input) value = dist(rng);
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "CPU comparison failed");
            ggml_backend_tensor_get(y, expected.data(), 0, expected.size() * sizeof(float));
            require(ggml_backend_graph_compute(npu, graph) == GGML_STATUS_SUCCESS, "NPU prefill failed");
            ggml_backend_tensor_get(y, measured.data(), 0, measured.size() * sizeof(float));
            double err = 0, sq = 0, edge_err = 0, edge_sq = 0;
            for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) {
                const size_t i = t * rows + r;
                const double e = std::pow(double(measured[i]) - expected[i], 2);
                const double s = std::pow(double(expected[i]), 2);
                err += e;
                sq += s;
                if (t == 0 || t == tokens - 1 || r == 0 || r == rows - 1) { edge_err += e; edge_sq += s; }
            }
            const double rms = std::sqrt(err / sq), edges = std::sqrt(edge_err / edge_sq);
            std::printf("pass=%d type=%s tokens=%d rows=%d rms_cpu=%.6f edge_rms_cpu=%.6f\n",
                        pass + 1, argv[3], tokens, rows, rms, edges);
            require(rms < 0.009 && edges < 0.009, "prefill output differential failed");
        }
        ggml_backend_buffer_free(data);
        ggml_free(ctx);
        ggml_backend_buffer_free(wb);
        ggml_free(wc);
        ggml_backend_free(npu);
        ggml_backend_free(cpu);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
