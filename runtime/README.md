# Phoenix XDNA 1 for Bionic 0.3.1

Phoenix 0.3.1 added compact Q4_0 prompt programs for LFM3-1.7B and Gemma 4 E2B IT. It accepts both the ordinary GGUF representation and Bionic’s AVX2-repacked weights. The NPU expands compact tiles during execution. Windows memory checks and per-stream budgets bound retained weights, with CPU fallback for matrices that would exceed the budget.

For a roughly 1,000-token document on a Ryzen 5 8645HS, median warm time to the first visible answer was 5.06 s on LFM3 and 2.90 s on Gemma, versus 8.17 s and 4.32 s on the paired CPU runs. Each warm figure uses four distinct requests on an already loaded instance. A separate generation workload reached 39.67 and 25.89 output tokens/s on Phoenix.

All ten document replies matched CPU exactly. Across all 70 paired replies, one LFM3 city-name response returned the right city with extra markup. The verifier exposes this difference under strict comparison and supports a narrow equivalent-city option.

## Install

Use Windows x64, a Phoenix or Hawk Point XDNA 1 device with a working AMD XRT driver, PowerShell 7 and the official Bionic CPU llama.cpp 2.80.1 AVX2 runtime. Install that CPU runtime in Bionic first. Keep GGUF models in your normal library.

Extract the ZIP to a new directory. In PowerShell 7, run the following from that directory.

```powershell
.\install.ps1
lms runtime select phoenix-xdna1-win-x86_64-avx2@0.3.1
```

Alternatively select **Phoenix XDNA 1 (Windows) 0.3.1** in Bionic's GGUF runtime settings. Unload and reload the model. To roll back, select the previous engine and reload the model.

## Verify and build

Run `.\verify.ps1 -ModelRoot <model-library>` with the installed test GGUFs. The verifier opens local Bionic instances and compares CPU and Phoenix responses. Add `-IncludeLargeModels` for the LFM3 and Gemma test suite. The known LFM3 city fixture accepts equivalent text with `-AllowEquivalentCityAnswer`. 

The release archive includes the Phoenix host and compiled NPU programs. To rebuild the host, install Visual Studio C++ Build Tools, the Windows SDK and a matching AMD XRT SDK. Run `.\build.ps1 -XrtSdk <sdk-directory>`. The third-party NPU toolchain is separate. The device sources and their attribution are under `programs`. Where included, `programs/source/build-programs.ps1 -Toolchain <toolchain-directory>` rebuilds the device programs with a compatible IRON toolchain layout.

The SHA-256 manifest covers every release file except the manifest itself. Review `THIRD_PARTY_NOTICES.md` and the accompanying license texts.
