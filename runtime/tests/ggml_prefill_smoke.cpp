// Exercise actual GGUF up-projection rows through the stock CPU and Phoenix.
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>

static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

int main(int argc, char ** argv) {
    try {
        const bool cpu_buffer = argc == 7 && !std::strcmp(argv[6], "cpu-buffer");
        const bool fill_cache = argc == 7 && !std::strcmp(argv[6], "fill-cache");
        require(argc == 6 || cpu_buffer || fill_cache,
                "expected CPU pack, Phoenix DLL, fixture, GGUF or raw weight, tokens, optional cpu-buffer/fill-cache");
        const bool smol_fixture = !std::strcmp(argv[3], "q5_0");
        const bool lfm_down = !std::strcmp(argv[3], "q4_0-lfm3-down");
        const bool lfm = lfm_down || !std::strcmp(argv[3], "q4_0-lfm3");
        const bool wide = !std::strcmp(argv[3], "q4_0-gemma-wide");
        const bool down = !std::strcmp(argv[3], "q4_0-gemma-down");
        require(smol_fixture || lfm || wide || down || !std::strcmp(argv[3], "q4_0-gemma"), "unknown projection fixture");
        const int rows = smol_fixture ? 2560 : lfm_down ? 2048 : lfm ? 7680 : wide ? 12288 : down ? 1536 : 6144;
        const int cols = smol_fixture ? 960 : lfm_down ? 7680 : lfm ? 2048 : down ? 6144 : 1536;
        const auto type = smol_fixture ? GGML_TYPE_Q5_0 : GGML_TYPE_Q4_0;
        const int tokens = std::stoi(argv[5]);
        require(tokens >= (smol_fixture ? 64 : 128) && tokens <= 512, "unsupported token count for projection fixture");
        ggml_backend_load_all_from_path(argv[1]);
        require(ggml_backend_load(argv[2]), "Phoenix DLL load failed");
        auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        auto * npu = ggml_backend_init_by_name("XDNA1", nullptr);
        require(cpu && npu, "backend init failed");
        ggml_backend_cpu_set_n_threads(cpu, 6);
        auto * wc = ggml_init({ggml_tensor_overhead(), nullptr, true});
        auto * weight = ggml_new_tensor_2d(wc, type, cols, rows);
        const char * tensor_name = wide ? "blk.15.ffn_up.weight" : (down || lfm_down) ? "blk.0.ffn_down.weight" : "blk.0.ffn_up.weight";
        ggml_set_name(weight, tensor_name);
        auto * probe_ctx = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
        auto * probe_input = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32, cols, 512);
        auto * probe = ggml_mul_mat(probe_ctx, weight, probe_input);
        auto * cpu_repack_weight = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_Q4_0, cols, rows);
        ggml_set_name(cpu_repack_weight, weight->name);
        auto * repack_probe = ggml_mul_mat(probe_ctx, cpu_repack_weight, probe_input);
        auto * marker = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(npu), 0);
        weight->buffer = cpu_repack_weight->buffer = marker;
        require(ggml_backend_supports_op(npu, probe) == smol_fixture, "incorrect canonical GGUF weight placement");
        require(!ggml_backend_supports_op(npu, repack_probe), "loader bypassed native CPU Q4_0 repacking");
        weight->buffer = cpu_repack_weight->buffer = nullptr;
        ggml_backend_buffer_free(marker);
        ggml_free(probe_ctx);
        auto * buft = ggml_backend_get_default_buffer_type(smol_fixture && !cpu_buffer ? npu : cpu);
        if (!smol_fixture && !cpu_buffer) {
            auto * dev = ggml_backend_get_device(cpu);
            auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
            require(extra, "CPU extra buffer inventory unavailable");
            for (auto p = extra(dev); p && *p; ++p) {
                if (!std::strcmp(ggml_backend_buft_name(*p), "CPU_REPACK")) buft = *p;
            }
            require(!std::strcmp(ggml_backend_buft_name(buft), "CPU_REPACK"), "Q4_0 CPU repacker unavailable");
        }
        auto * wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, buft);
        require(wb, "GGUF weight allocation failed");
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
        const auto * budget = std::getenv("PHOENIX_XDNA_PROMPT_CACHE_MIB");
        const bool lower = budget && !std::strcmp(budget, "384");
        const bool standard = lower || (budget && !std::strcmp(budget, "700"));
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        const bool full = lfm && !standard && GlobalMemoryStatusEx(&status) &&
                          status.ullTotalPhys >= (uint64_t(16) << 30) && status.ullAvailPhys >= (uint64_t(6) << 30);
        require(data, "tensor allocation failed");
        if (lfm_down && !full) {
            require(!ggml_backend_supports_op(npu, y), "low-memory profile accepted the extra LFM down projection");
            std::puts("Low-memory LFM3 profile correctly retained the CPU down projection.");
            ggml_backend_buffer_free(data);
            ggml_free(ctx);
            ggml_backend_buffer_free(wb);
            ggml_free(wc);
            ggml_backend_free(npu);
            ggml_backend_free(cpu);
            return 0;
        }
        require(ggml_backend_supports_op(npu, y), "NPU rejected projection");
        if (!smol_fixture) {
            const size_t first_excluded = lfm ? (full ? 24 : lower ? 10 : 19) : down ? (lower ? 5 : 15) :
                                          wide ? (lower ? 16 : 23) : (lower ? 17 : 32);
            const char * suffix = down || lfm_down ? ".ffn_down.weight" : ".ffn_gate.weight";
            ggml_set_name(weight, ("blk." + std::to_string(first_excluded - 1) + suffix).c_str());
            require(ggml_backend_supports_op(npu, y), "last budgeted projection was rejected");
            ggml_set_name(weight, ("blk." + std::to_string(first_excluded) + suffix).c_str());
            require(!ggml_backend_supports_op(npu, y), "projection exceeds cache admission budget");
            ggml_set_name(weight, tensor_name);
        }
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
        if (fill_cache) {
            const auto * budget = std::getenv("PHOENIX_XDNA_PROMPT_CACHE_MIB");
            require(wide && budget && !std::strcmp(budget, "384"), "fill-cache expects the wide fixture and 384 MiB budget");
            const size_t entry = ggml_nbytes(weight) + size_t(rows) * cols * 20 / 32;
            const size_t allowed = (size_t(384) << 20) / entry;
            auto * extra = ggml_init({allowed * ggml_tensor_overhead(), nullptr, true});
            std::vector<ggml_tensor *> duplicates;
            for (size_t i = 0; i < allowed; ++i) {
                // Independent copies simulate many model weights with eligible
                // names, beyond the mixed-width graph placement estimate.
                auto * w = ggml_new_tensor_2d(extra, type, cols, rows);
                ggml_set_name(w, tensor_name);
                duplicates.push_back(w);
            }
            auto * buffers = ggml_backend_alloc_ctx_tensors_from_buft(extra, buft);
            require(buffers, "could not allocate budget regression fixtures");
            ggml_backend_buffer_set_usage(buffers, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> result(measured.size());
            for (size_t i = 0; i < duplicates.size(); ++i) {
                auto * w = duplicates[i];
                ggml_backend_tensor_set(w, codes.data(), 0, codes.size());
                y->src[0] = w;
                require(ggml_backend_graph_compute(npu, graph) == GGML_STATUS_SUCCESS, "bounded prefill failed");
                ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
                require(result == (i + 1 == duplicates.size() ? expected : measured),
                        "incorrect hardware selection at the cache capacity boundary");
            }
            std::printf("Kept %zu compact matrices under 384 MiB; next projection reproduced exact CPU output.\n", allowed);
            y->src[0] = weight;
            ggml_backend_buffer_free(buffers);
            ggml_free(extra);
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
