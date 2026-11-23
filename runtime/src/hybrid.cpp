// Bionic protocol and LLama graph execution remain in the pinned Vulkan server.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <initguid.h>
#include <dxcore.h>
#include <wrl/client.h>
#include <io.h>
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

__declspec(dllimport) int llama_server(int, char **);
namespace fs = std::filesystem;
using Microsoft::WRL::ComPtr;

static std::string setting(const fs::path & dir) {
    const auto path = dir / "hybrid-mode.txt";
    if (!fs::exists(path)) return "cpu-npu";
    std::ifstream f(path);
    std::string mode;
    if (!std::getline(f, mode) ||
        (mode != "radeon" && mode != "split-head" && mode != "split-head-bf16" && mode != "prompt-ffn" &&
         mode != "cpu-npu" && mode != "cpu-prompt" && mode != "cpu-prompt-resident")) {
        throw std::runtime_error("Unrecognized Phoenix placement in hybrid-mode.txt");
    }
    return mode;
}

// Limit this experimental pack to a unique integrated AMD graphics adapter.
// The AMD Vulkan ICD plus its first visible device then selects that physical GPU.
static unsigned amd_igpu() {
    ComPtr<IDXCoreAdapterFactory> factory;
    ComPtr<IDXCoreAdapterList> list;
    const GUID graphics = DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS;
    if (FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->CreateAdapterList(1, &graphics, IID_PPV_ARGS(&list)))) {
        throw std::runtime_error("DXCore graphics enumeration failed");
    }
    unsigned found = 0;
    for (uint32_t i = 0; i < list->GetAdapterCount(); ++i) {
        ComPtr<IDXCoreAdapter> adapter;
        DXCoreHardwareID id{};
        bool integrated = false;
        if (FAILED(list->GetAdapter(i, IID_PPV_ARGS(&adapter))) ||
            FAILED(adapter->GetProperty(DXCoreAdapterProperty::HardwareID, &id))) {
            throw std::runtime_error("DXCore adapter identification failed");
        }
        if (id.vendorID != 0x1002) continue;
        if (found || FAILED(adapter->GetProperty(DXCoreAdapterProperty::IsIntegrated,&integrated)) || !integrated)
            throw std::runtime_error("Experimental Vulkan route needs exactly one AMD graphics device, an iGPU");
        found = id.deviceID;
    }
    if (!found) throw std::runtime_error("No integrated AMD graphics device");
    return found;
}

static void configure_driver(const fs::path & dir) {
    std::wifstream f(dir / "amd-icd.txt");
    std::wstring driver;
    if (!std::getline(f,driver) || !fs::is_regular_file(driver) || fs::path(driver).filename() != L"amd-vulkan64.json")
        throw std::runtime_error("Install the AMD Vulkan driver and re-run the pack installer");
    if (_wputenv_s(L"VK_DRIVER_FILES",driver.c_str()) || _wputenv_s(L"VK_ICD_FILENAMES",driver.c_str()) ||
        _wputenv_s(L"VK_LOADER_DRIVERS_SELECT",L"amd-vulkan64.json") || _wputenv_s(L"VK_LOADER_DRIVERS_DISABLE",L"") ||
        _wputenv_s(L"DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1",L"1") || _putenv_s("GGML_VK_VISIBLE_DEVICE_UUIDS","") ||
        _putenv_s("GGML_VK_VISIBLE_DEVICES","0")) throw std::runtime_error("Vulkan driver selection failed");
}

int main(int argc, char ** argv) {
    try {
        wchar_t executable[32768];
        const auto n = GetModuleFileNameW(nullptr,executable,32768);
        if (!n || n >= 32768) throw std::runtime_error("Cannot identify runtime folder");
        const auto dir = fs::path(executable).parent_path();
        const auto mode = setting(dir);
        const bool graphics = mode.rfind("cpu-", 0) != 0;
        const bool prompt = mode == "prompt-ffn" || mode == "cpu-prompt" || mode == "cpu-prompt-resident";
        unsigned device_id = 0;
        if (graphics) {
            configure_driver(dir);
            device_id = amd_igpu();
        }
        const auto trace = dir / "dispatch-test.log";
        const bool tracing = fs::is_regular_file(trace);

        std::vector<std::string> args = {argv[0]};
        const std::vector<std::string> controlled = {"--device","-dev","--split-mode","-sm","--main-gpu","-mg","--tensor-split","-ts",
            "--n-gpu-layers","--gpu-layers","-ngl","--override-tensor","-ot"};
        bool model = false;
        for (int i=1;i<argc;++i) {
            const std::string opt(argv[i]);
            const auto equal = opt.find('=');
            const auto key = opt.substr(0,equal);
            if (key == "--model" || key == "-m") model = true;
            if (std::find(controlled.begin(),controlled.end(),key) != controlled.end() ||
                (tracing && (key=="--log-file" || key=="--log-verbosity" || key=="--verbosity" || key=="-lv"))) {
                if (equal == std::string::npos && ++i == argc) throw std::runtime_error("Missing runtime argument value");
                continue;
            }
            if (tracing && key=="--log-disable") continue;
            args.push_back(opt);
        }

        const std::string stem = "hybrid-" + std::to_string(GetCurrentProcessId());
        if (model && tracing) {
            const auto path = dir / (stem + "-native.log");
            FILE * err = nullptr;
            if (_wfreopen_s(&err,path.c_str(),L"wb",stderr) || !err ||
                !SetStdHandle(STD_ERROR_HANDLE,reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(err))))) {
                throw std::runtime_error("Could not establish native trace");
            }
        }
        if (mode != "radeon") {
            if (_putenv_s("PHOENIX_XDNA_PLACEMENT",prompt ? "prompt-only" : "split-head") ||
                _putenv_s("PHOENIX_XDNA_Q1",mode=="split-head-bf16" ? "0" : "1") ||
                _putenv_s("PHOENIX_XDNA_RESIDENT_BF16",mode=="cpu-prompt-resident" ? "1" : "0") ||
                _putenv_s("PHOENIX_XDNA_HEAD_TILES","1") || _putenv_s("PHOENIX_XDNA_PREFILL","1") ||
                (tracing && _wputenv_s(L"PHOENIX_XDNA_TRACE",trace.c_str()))) throw std::runtime_error("XDNA configuration failed");
            const auto plugin = dir / "ggml-phoenix.dll";
            auto reg = ggml_backend_load(plugin.string().c_str());
            if (!reg || ggml_backend_reg_dev_count(reg)!=1 ||
                std::string(ggml_backend_buft_name(ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg,0)))) != "Phoenix_GGUF")
                throw std::runtime_error("Phoenix XDNA plugin ABI or buffer registration failed");
        }

        if (model) {
            args.insert(args.end(),{"--device",graphics ? "Vulkan0" : "none","--n-gpu-layers",graphics ? "all" : "0","--split-mode","none"});
            if (graphics) args.insert(args.end(),{"--main-gpu","0"});
            if (mode != "radeon") args.push_back("--op-offload");
            // Tied vocabulary heads use token_embd.weight. On Bonsai, output.weight is distinct.
            if (mode=="split-head" || mode=="split-head-bf16") args.insert(args.end(),{"--override-tensor","^(output|token_embd)[.]weight$=Phoenix_GGUF"});
            if (mode=="prompt-ffn") args.insert(args.end(),{"--override-tensor","^blk[.]0[.]ffn_gate[.]weight$=Phoenix_GGUF"});
            if (tracing) {
                std::ofstream f(trace,std::ios::app);
                f << "[Phoenix Hybrid] pid=" << GetCurrentProcessId() << " mode=" << mode
                  << " vulkan=" << (graphics ? "Vulkan0" : "none") << " amd_igpu=0x" << std::hex << device_id
                  << " gpu_layers=" << (graphics ? "all" : "0") << "\n";
                if (!f) throw std::runtime_error("Could not write routing receipt");
                args.insert(args.end(),{"--log-file",(dir/(stem+".log")).string(),"--log-verbosity","4"});
            }
        }
        std::vector<char *> av;
        for (auto & s: args) av.push_back(s.data());
        av.push_back(nullptr);
        return llama_server(static_cast<int>(args.size()),av.data());
    } catch (const std::exception & e) {
        std::fprintf(stderr,"Phoenix Hybrid: %s\n", e.what());
        return 1;
    }
}
