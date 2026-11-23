// Keep Bionic's llama-server entry point, argument parser and process lifetime.
// The only added backend runs eligible GGML operations on the Phoenix NPU.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
        const auto trace_file = directory / L"dispatch-test.log";
        const DWORD flags = GetFileAttributesW(trace_file.c_str());
        if (!_wgetenv(L"PHOENIX_XDNA_TRACE") && flags != INVALID_FILE_ATTRIBUTES &&
            (flags & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0) {
            if (_wputenv_s(L"PHOENIX_XDNA_TRACE", trace_file.c_str()) != 0) {
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

        // ACCEL devices participate alongside the CPU. Repacking would turn the
        // source GGUF rows into CPU-specific blocks that this backend cannot read.
        // These switches apply only to this explicitly selected runtime.
        char no_repack[] = "--no-repack";
        char op_offload[] = "--op-offload";
        std::vector<char *> args(argv, argv + argc);
        args.push_back(no_repack);
        args.push_back(op_offload);
        args.push_back(nullptr);
        return llama_server(static_cast<int>(args.size() - 1), args.data());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Phoenix XDNA1: %s\n", error.what());
        return 1;
    }
}
