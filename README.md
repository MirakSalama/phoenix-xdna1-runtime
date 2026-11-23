# Phoenix XDNA 1 Runtime

Run local GGUF language models on a first-generation AMD Ryzen AI NPU in Bionic on Windows. Phoenix adds a selectable llama.cpp engine alongside the CPU, CUDA and Vulkan engines. Open the same GGUF you already use. The runtime sends eligible operations to the NPU and keeps the rest on the CPU.

**Phoenix is most useful for reading long prompts while the Radeon is needed for graphics.** On a Ryzen 5 8645HS, Phoenix 0.4.0 shortened the warm wait for the first visible answer by **47% on LFM3-1.7B** and **35% on Gemma 4 E2B** versus Bionic’s stock CPU engine. [Download the latest stable Windows runtime](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/latest)

## Install and run

1. Use Windows x64, PowerShell 7 and an AMD Phoenix or Hawk Point processor with an **XDNA 1 NPU** and a working Windows XRT driver. The hardware measurements below come from one Ryzen 5 8645HS laptop.
2. In Bionic, install **CPU llama.cpp (Windows) 2.80.1** for GGUF. Phoenix uses native libraries from this exact local pack. Bionic was the tested host. The project uses LM Studio’s `lms` runtime commands. [Compatibility lock](runtime/compatibility.json) · [LM Studio runtime CLI](https://lmstudio.ai/docs/cli/runtime/runtime)
3. Download the **runtime ZIP** under **Assets** on the latest stable release and extract it. Open PowerShell 7 in the extracted directory. The commands below show version 0.4.0. The first checks compatibility. Installation stages a versioned pack and checks a numerical result on the NPU. [Latest release assets](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/latest)

```powershell
.\install.ps1 -CheckOnly
.\install.ps1
lms runtime select phoenix-xdna1-win-x86_64-avx2@0.4.0
```

4. Load or reload your GGUF in Bionic and start a chat. You can also choose **Settings → Runtime → GGUF → Phoenix XDNA 1 (Windows) 0.4.0**. To switch back, select the previous engine and reload the model.

## What gets faster

The document test asked for a pickup code buried in nine records, roughly 1,000 input tokens. Each engine used the same local weights and answered one request at a time. The table shows the median time to **first visible answer text** across four different documents after the first request on a loaded model. The default Phoenix configuration uses compact weights and shares eligible vocabulary work with the CPU.

| Model and GGUF | Stock CPU | Phoenix 0.4.0 | Radeon 760M Vulkan |
| --- | ---: | ---: | ---: |
| LFM3-1.7B QAD Q4_0 | 7.93 s | **4.24 s** | 0.94 s |
| Gemma 4 E2B IT Q4_0 | 3.85 s | **2.52 s** | 0.55 s |

Radeon Vulkan delivered the quickest answers on this laptop. Generating output tokens also favored the GPU. For example, on a separate LFM3 40-token task, the Radeon generated 63.44 output tokens/s, CPU 43.82, and default Phoenix 41.50. Phoenix’s measured advantage over CPU is concentrated in processing the input. [Release measurements and test conditions](runtime/README.md)

A separate test ran a fixed 1280 × 720 offscreen Direct3D 11 graphics workload on the Radeon during chat. On LFM3, the 95th-percentile frame time was **2.91 ms** with Phoenix inference and **25.19 ms** with Radeon inference. On Gemma it was **2.37 ms** and **16.96 ms**, respectively. Radeon still answered sooner. These measurements describe GPU-completed frames in the synthetic test. Games, displays and workloads can behave differently.

![Document latency and graphics frame times](docs/benchmarks.png)

For sample sizes, timing boundaries and the decision to retain 0.4.0 after a startup experiment, see the [benchmark methodology and results](docs/benchmarks.md).

## Model coverage and settings

- **LFM3-1.7B QAD Q4_0 and Gemma 4 E2B IT Q4_0.** Supported feed-forward operations run on the NPU during prompt processing. Phoenix reads the original block scales and handles Bionic’s AVX2-repacked weights.
- **SmolLM2-360M Instruct Q4_K_M.** Suitable feed-forward projections can use the NPU. **LFM2.5-230M Q4_K_M** uses CPU prompt processing and can share eligible generation work with the NPU.
- **Other GGUFs.** Bionic’s pinned llama.cpp build determines which files load. Offload depends on tensor shape, weight type and prompt size. A model that opens successfully can still run entirely on CPU.

To let the CPU handle all generation and use the NPU for eligible prompts, run `.\set-mode.ps1 -Mode prompt-only`. For repeated long prompts on machines with RAM to spare, `.\set-mode.ps1 -ResidentBF16` allows Phoenix to expand eligible weights in memory. This requires at least 16 GiB of physical RAM and 8 GiB available at startup. Restore both defaults with `.\set-mode.ps1 -Mode split-head -Compact`. Reload the model after changing settings. [Memory limits and supported operations](runtime/README.md)

## Verify or troubleshoot

- **Installer reports a base-library mismatch.** Check the installed Windows CPU engine with `lms runtime ls` and rerun `.\install.ps1 -CheckOnly`. The compatibility lock records the exact binaries used for this release.
- **The pack is selected but the NPU graph is quiet.** Try a supported model with a long prompt. The supplied verifier checks completed hardware operations through Bionic and compares responses with CPU. It needs Python 3 and the corresponding GGUFs in your library. In the full default release suite, 69 of 70 replies matched CPU text and token counts exactly. One LFM3 city answer repeated “Paris” with extra markup. All ten document replies matched.
- **To reproduce the release checks,** run the following in the extracted 0.4.0 directory. The city-answer flag allows the documented variation and records raw differences. The verifier temporarily selects the test engines and restores the previous selection. [Verification script](runtime/verify.ps1)

```powershell
$modelLibrary = Read-Host 'Path to your Bionic models folder'
.\verify.ps1 -ModelRoot $modelLibrary -IncludeLargeModels -VerifyResidentBF16 -AllowEquivalentCityAnswer
```

## Releases and project background

| Version | Milestone |
| --- | --- |
| [0.1.0](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.1.0) | First Bionic replies with verified XDNA 1 execution on ordinary GGUFs. |
| [0.2.0](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.2.0) | Cached prepared weights, shared CPU/NPU generation and accelerated selected SmolLM2 prompts. |
| [0.3.1](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.3.1) | Compact four-bit prompt operations on LFM3 and Gemma 4. |
| [0.4.0](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.4.0) | Wider prompt tiles, overlapping input preparation and selectable generation placement. |

This runtime grew out of **Phoenix AI**, a separate application with its own releases. Abhiram Jayasankar’s **Ryzen AI NPU Lab**, including its original LFM2.5 NPU engine, helped start the Phoenix AI project. The runtime builds on llama.cpp’s GGML operations and uses AMD XRT to submit work to the NPU. [Phoenix AI](https://github.com/MirakSalama/Phoenix-AI) · [Ryzen AI NPU Lab](https://github.com/AbhiramJayasankar/ryzenai-npu-lab) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [AMD XRT](https://github.com/Xilinx/XRT)

[Release history](CHANGELOG.md) · [Publishing a verified release](publish/MIRRORING.md) · [Third-party notices](runtime/THIRD_PARTY_NOTICES.md)
