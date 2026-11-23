# Phoenix XDNA 1 Runtime changelog

The versions below are separately installable Windows runtime packs. Each tag pins the matching `runtime/` source snapshot. The table of releases on GitHub includes subsequent stable versions. [All releases](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases)

## 0.5.0

- Added one-command 16-core XDNA 1 scoring for the first 3,072 rows of supported 5,120-wide Q1_0 output heads. Bonsai uses its original packed GGUF weights. The native slice takes 2.11 MiB of device weight storage. A standalone 40-call trial on a Ryzen 5 8645HS measured a 0.764 ms median for this operation.
- Added selectable Radeon transformer layers with a split CPU/XDNA head, an expanded-BF16 comparison, prompt-gate placement and a Radeon-only control. CPU and XDNA remain the default placement. Reload a model after switching modes.
- Pinned Bionic Vulkan llama.cpp 2.80.1 and Phoenix 0.4.0 inputs. The pack includes kernel source, a clean-build verifier and exact device payload digests. [0.5.0 release](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.5.0)

## 0.4.0

- Added 256-token prompt tiles for five LFM3 and Gemma projection shapes. Input staging can overlap the preceding tile’s NPU execution.
- Added prompt-only NPU placement with CPU generation, plus optional resident BF16 weights for systems with enough available RAM.
- Recorded warm first visible answers of 4.24 seconds for LFM3-1.7B and 2.52 seconds for Gemma 4 E2B on the document test. Stock CPU took 7.93 and 3.85 seconds. [Release and results](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.4.0)

## 0.3.1

- Added compact four-bit prompt operations for supported LFM3-1.7B and Gemma 4 E2B GGUFs, including Bionic’s CPU-repacked weight layout.
- Added bounded per-stream caches and verified eligible projection shapes with real GGUF weights.
- On the same document workload, version 0.3.1 delivered the first visible answer in 5.06 seconds on LFM3 and 2.90 seconds on Gemma. These are warm medians. [Release and results](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.3.1)

## 0.2.0

- Cached prepared weights and let CPU and NPU compute separate parts of the vocabulary score vector concurrently.
- Added NPU prompt acceleration for compatible SmolLM2 feed-forward layers. Introduced side-by-side installation checks and a repeatable native and Bionic verification suite. [Release and results](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.2.0)

## 0.1.0

- Added a selectable Windows Bionic runtime using AMD XRT for eligible GGML operations during generation.
- Verified NPU execution and real replies using original LFM2.5-230M and SmolLM2-360M GGUFs on a Ryzen 5 8645HS. [Release and results](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/tag/v0.1.0)
