// Keep Bionic's llama-server entry point, argument parser and process lifetime.
// The only added backend runs eligible GGML operations on the Phoenix NPU.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ggml-backend.h"

#include <cstdio>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

__declspec(dllimport) int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
    try {
        wchar_t executable[32768];
        const DWORD length = GetModuleFileNameW(nullptr, executable, 32768);
        if (length == 0 || length >= 32768) {
            std::fprintf(stderr, "Phoenix XDNA1: unable to locate the runtime directory.\n");
            return 1;
        }
        const auto directory = std::filesystem::path(executable).parent_path();
        const DWORD storage_flags = GetFileAttributesW((directory / L"phoenix-resident-bf16").c_str());
        if (!std::getenv("PHOENIX_XDNA_RESIDENT_BF16") && storage_flags != INVALID_FILE_ATTRIBUTES &&
            (storage_flags & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0 &&
            _putenv_s("PHOENIX_XDNA_RESIDENT_BF16", "1") != 0) return 1;
        const DWORD mode_flags = GetFileAttributesW((directory / L"phoenix-prompt-only").c_str());
        if (!std::getenv("PHOENIX_XDNA_PLACEMENT") && mode_flags != INVALID_FILE_ATTRIBUTES &&
            (mode_flags & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0 &&
            _putenv_s("PHOENIX_XDNA_PLACEMENT", "prompt-only") != 0) return 1;
        const auto trace_file = directory / L"dispatch-test.log";
        const DWORD flags = GetFileAttributesW(trace_file.c_str());
        if (flags != INVALID_FILE_ATTRIBUTES &&
            (flags & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0) {
            // The performance harness owns this file. Only known placement modes
            // may be requested for a single instrumented engine process.
            std::ifstream input(trace_file, std::ios::binary);
            char header[64]{};
            for (int i = 0; i < 4 && input.getline(header, sizeof header); ++i) {
                const std::string line(header);
                if (line.rfind("placement=", 0) == 0) {
                    for (const char * mode : {"all", "dense", "head", "split-head", "prompt-only"}) {
                        if (line.substr(10) == mode && _putenv_s("PHOENIX_XDNA_PLACEMENT", mode) != 0) return 1;
                    }
                } else if (line.rfind("tiles=", 0) == 0) {
                    int tiles = 0;
                    const char * end = header + line.size();
                    const auto parsed = std::from_chars(header + 6, end, tiles);
                    if (parsed.ec == std::errc{} && parsed.ptr == end && tiles >= 1 && tiles <= 32 &&
                        _putenv_s("PHOENIX_XDNA_HEAD_TILES", header + 6) != 0) return 1;
                } else if (line == "prefill=1" || line == "prefill=0") {
                    if (_putenv_s("PHOENIX_XDNA_PREFILL", header + 8) != 0) return 1;
                } else break;
            }
            input.close();
            const char * mode = std::getenv("PHOENIX_XDNA_PLACEMENT");
            std::ofstream output(trace_file, std::ios::binary | std::ios::app);
            output << "[Phoenix XDNA1] decode_device=" << (mode && !std::strcmp(mode, "prompt-only") ? "cpu" : "npu-enabled") << '\n';
            output.close();
            if (!output) return 1;
            if (!_wgetenv(L"PHOENIX_XDNA_TRACE") && _wputenv_s(L"PHOENIX_XDNA_TRACE", trace_file.c_str()) != 0) {
                std::fprintf(stderr, "Phoenix XDNA1: failed to enable the dispatch trace.\n");
                return 1;
            }
        }

        const auto plugin = directory / L"ggml-phoenix.dll";
        const auto registration = ggml_backend_load(plugin.string().c_str());
        if (!registration || ggml_backend_reg_dev_count(registration) == 0) {
            std::fprintf(stderr, "Phoenix XDNA1: failed to register the NPU backend.\n");
            return 1;
        }
        if (argc == 2 && std::strcmp(argv[1], "--phoenix-self-test") == 0) {
            using SelfTest = int (*)();
            const auto module = GetModuleHandleW(plugin.c_str());
            const auto test = module ? reinterpret_cast<SelfTest>(GetProcAddress(module, "phoenix_xdna_self_test")) : nullptr;
            return test ? test() : 1;
        }

        // Normal chat preserves CPU weight repacking. Only the layer-wide
        // diagnostic placements need access to the original rows of every layer.
        char no_repack[] = "--no-repack";
        char op_offload[] = "--op-offload";
        std::vector<char *> args(argv, argv + argc);
        const char * placement = std::getenv("PHOENIX_XDNA_PLACEMENT");
        if (placement && (!std::strcmp(placement, "dense") || !std::strcmp(placement, "all"))) {
            args.push_back(no_repack);
        }
        args.push_back(op_offload);
        args.push_back(nullptr);
        return llama_server(static_cast<int>(args.size() - 1), args.data());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Phoenix XDNA1: %s\n", error.what());
        return 1;
    }
}
