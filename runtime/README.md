# Phoenix XDNA 1 for Bionic 0.1.0

Phoenix 0.1.0 introduced local GGUF chat on first-generation AMD XDNA hardware in Bionic. Bionic loads the model and runs tokenization, attention, prompt processing and sampling. During generation, eligible dense matrix-vector products run on the NPU using the weights from the selected GGUF.

On a Ryzen 5 8645HS, the original LFM2.5-230M and SmolLM2-360M four-bit GGUFs returned the same eight-token reply as CPU. One request after model load took 2.453 s and 5.328 s respectively on Phoenix, versus 0.078 s and 0.203 s on CPU. Traces recorded completed NPU work during both responses.

This early implementation converts and stages weights on repeated calls. It provides a baseline for later releases.

## Install

Use Windows x64, a Phoenix or Hawk Point XDNA 1 device with a working AMD XRT driver, PowerShell 7 and the official Bionic CPU llama.cpp 2.80.1 AVX2 runtime. Install that CPU runtime in Bionic first. Keep GGUF models in your normal library.

Extract the ZIP to a new directory. In PowerShell 7, run the following from that directory.

```powershell
.\install.ps1
lms runtime select phoenix-xdna1-win-x86_64-avx2@0.1.0
```

Alternatively select **Phoenix XDNA 1 (Windows) 0.1.0** in Bionic's GGUF runtime settings. Unload and reload the model. To roll back, select the previous engine and reload the model.

## Verify and build

Use the local Bionic chat test in `tests/bionic_chat.py` to compare CPU and Phoenix on the installed test GGUFs. Supply a fresh filename using `--report`.

The release archive includes the Phoenix host and compiled NPU programs. To rebuild the host, install Visual Studio C++ Build Tools, the Windows SDK and a matching AMD XRT SDK. Run `.\build.ps1 -XrtSdk <sdk-directory>`. The third-party NPU toolchain is separate. The device sources and their attribution are under `programs`. Where included, `programs/source/build-programs.ps1 -Toolchain <toolchain-directory>` rebuilds the device programs with a compatible IRON toolchain layout.

The SHA-256 manifest covers every release file except the manifest itself. Review `THIRD_PARTY_NOTICES.md` and the accompanying license texts.
