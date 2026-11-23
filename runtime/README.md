# Phoenix XDNA 1 for Bionic 0.4.0

Phoenix 0.4.0 adds 256-token compact prompt programs for five LFM3 and Gemma matrix shapes. On fresh multi-tile inputs it stages the next activation while the NPU processes the current one. The default uses compact four-bit weights, with the CPU and NPU sharing eligible generation work. Optional modes keep more weights expanded in RAM or leave all generation on the CPU.

On a Ryzen 5 8645HS, warm time to the first visible answer on a roughly 1,000-token document was 4.24 s on LFM3-1.7B and 2.52 s on Gemma 4 E2B IT. The paired stock CPU times were 7.93 s and 3.85 s. Radeon 760M Vulkan returned answers in 0.94 s and 0.55 s on the same task. Reported warm medians use four different requests after the first on a loaded model.

A separate Gemma Q4_0 prompt-only benchmark reached 557 input tokens/s at 512 tokens against 362 on the bracketing CPU runs. All ten Bionic document responses matched CPU text and token counts. Across the 70-pair default suite, 69 matched exactly. One LFM3 response repeated the correct city with extra markup. Ordinary GGUF compatibility follows the base runtime. Acceleration depends on model shapes, batch and quantization.

Use `.\set-mode.ps1 -Mode prompt-only` to keep generation on the CPU, or `.\set-mode.ps1 -Mode split-head` to restore the default. Use `-ResidentBF16` to opt into expanded weights when memory permits, and `-Compact` to restore compact storage. Reload the model after changing a setting.

## Install

Use Windows x64, a Phoenix or Hawk Point XDNA 1 device with a working AMD XRT driver, PowerShell 7 and the official Bionic CPU llama.cpp 2.80.1 AVX2 runtime. Install that CPU runtime in Bionic first. Keep GGUF models in your normal library.

Extract the ZIP to a new directory. In PowerShell 7, run the following from that directory.

```powershell
.\install.ps1
lms runtime select phoenix-xdna1-win-x86_64-avx2@0.4.0
```

Alternatively select **Phoenix XDNA 1 (Windows) 0.4.0** in Bionic's GGUF runtime settings. Unload and reload the model. To roll back, select the previous engine and reload the model.

## Verify and build

Run `.\verify.ps1 -ModelRoot <model-library>` with the installed test GGUFs. The verifier opens local Bionic instances and compares CPU and Phoenix responses. Add `-IncludeLargeModels` for the LFM3 and Gemma test suite. The known LFM3 city fixture accepts equivalent text with `-AllowEquivalentCityAnswer`. 

The release archive includes the Phoenix host and compiled NPU programs. To rebuild the host, install Visual Studio C++ Build Tools, the Windows SDK and a matching AMD XRT SDK. Run `.\build.ps1 -XrtSdk <sdk-directory>`. The third-party NPU toolchain is separate. The device sources and their attribution are under `programs`. Where included, `programs/source/build-programs.ps1 -Toolchain <toolchain-directory>` rebuilds the device programs with a compatible IRON toolchain layout.

The SHA-256 manifest covers every release file except the manifest itself. Review `THIRD_PARTY_NOTICES.md` and the accompanying license texts.
