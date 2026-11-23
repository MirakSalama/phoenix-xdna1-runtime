# Phoenix XDNA 1 for Bionic 0.2.0

Phoenix 0.2.0 retains prepared weights in bounded per-instance buffers. The CPU and NPU compute distinct rows of the vocabulary projection concurrently. On supported prompts of 96–512 tokens, the NPU also processes 960-by-2,560 feed-forward projections, including those used by SmolLM2-360M.

On a Ryzen 5 8645HS, a warm eight-token chat reply reached 186.30 output tokens/s on LFM2.5-230M and 98.81 on SmolLM2-360M, versus 3.23 and 1.39 on Phoenix 0.1.0. The corresponding stock CPU rates were 214.63 and 112.75. For a 266–267-token SmolLM2 prompt, median warm time to first token fell from 799 ms on CPU to 595 ms on Phoenix. Four warm requests followed the first request on each loaded model.

All 40 recorded release replies matched the paired CPU text and token counts. Model loading, unsupported operations and sampling continue to use the pinned Bionic CPU runtime.

## Install

Use Windows x64, a Phoenix or Hawk Point XDNA 1 device with a working AMD XRT driver, PowerShell 7 and the official Bionic CPU llama.cpp 2.80.1 AVX2 runtime. Install that CPU runtime in Bionic first. Keep GGUF models in your normal library.

Extract the ZIP to a new directory. In PowerShell 7, run the following from that directory.

```powershell
.\install.ps1
lms runtime select phoenix-xdna1-win-x86_64-avx2@0.2.0
```

Alternatively select **Phoenix XDNA 1 (Windows) 0.2.0** in Bionic's GGUF runtime settings. Unload and reload the model. To roll back, select the previous engine and reload the model.

## Verify and build

Run `.\verify.ps1 -ModelRoot <model-library>` with the installed test GGUFs. The verifier opens local Bionic instances and compares CPU and Phoenix responses. 

The release archive includes the Phoenix host and compiled NPU programs. To rebuild the host, install Visual Studio C++ Build Tools, the Windows SDK and a matching AMD XRT SDK. Run `.\build.ps1 -XrtSdk <sdk-directory>`. The third-party NPU toolchain is separate. The device sources and their attribution are under `programs`. Where included, `programs/source/build-programs.ps1 -Toolchain <toolchain-directory>` rebuilds the device programs with a compatible IRON toolchain layout.

The SHA-256 manifest covers every release file except the manifest itself. Review `THIRD_PARTY_NOTICES.md` and the accompanying license texts.
