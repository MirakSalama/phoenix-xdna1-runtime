# Phoenix XDNA 1 v0.4.0 — Faster document prompts on Windows Ryzen AI

On a Ryzen 5 8645HS, Phoenix now returns the first visible text for a roughly 1,000-token document in **4.24 seconds on LFM3-1.7B QAD Q4_0** and **2.52 seconds on Gemma 4 E2B IT Q4_0**. Stock CPU took 7.93 and 3.85 seconds, respectively. The default configuration cut the warm wait by **47% and 35%**, measured over four different documents per model after the first request on a loaded instance.

- **Wider prompt tiles.** Eligible LFM3 and Gemma operations can process 256 tokens in one submission. The host can prepare the next tile while the NPU processes the current one.
- **Independent placement and memory settings.** Compact weights and shared CPU/NPU vocabulary work remain the defaults. Use `.\set-mode.ps1 -Mode prompt-only` to run generation on CPU, or `.\set-mode.ps1 -ResidentBF16` to prepare eligible weights in extra RAM. Reload the model to apply the change.
- **Verification.** The native tests cover both tile widths and ordinary and Bionic-repacked weights. All ten document replies matched stock CPU text and token counts. Across 70 default comparisons, one LFM3 city answer repeated “Paris” with extra markup.

Extract the runtime ZIP from **Assets**, run `.\install.ps1 -CheckOnly` followed by `.\install.ps1`, and select **Phoenix XDNA 1 (Windows) 0.4.0** in Bionic. Bionic’s Windows x64 CPU llama.cpp 2.80.1 pack and a working XDNA 1 XRT driver are required. [Versioned source, full results and verification](https://github.com/MirakSalama/phoenix-xdna1-runtime/tree/v0.4.0/runtime)

[Download the Windows runtime ZIP](https://github.com/MirakSalama/phoenix-xdna1-runtime/releases/download/v0.4.0/Phoenix-XDNA1-0.4.0.zip) · [Runtime sources and installation](runtime/README.md) · [Third-party notices](runtime/THIRD_PARTY_NOTICES.md)
