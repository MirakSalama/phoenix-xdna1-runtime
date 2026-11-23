# Phoenix XDNA 1 0.5.0 for Bionic

Phoenix 0.5.0 adds a native packed one-bit Bonsai 27B output kernel and selectable CPU, XDNA 1 and integrated Radeon placements. Bionic loads the original GGUF.

The 16-core kernel scores 3,072 of Bonsai's 248,320 vocabulary entries per eligible generation step. The GPU and CPU participation depend on placement. The packed slice takes 2.11 MiB of weight storage versus 30 MiB in the expanded BF16 implementation. On a Ryzen 5 8645HS, a 40-call standalone trial measured 0.764 ms median for this slice. The published kernel passed reference comparisons using real model weights and synthetic bit patterns. Whole-chat performance also depends on the rest of the vocabulary, the transformer layers, prompt length and memory contention.

In Bionic, four 40-output-token counting requests in each split-head mode measured 4.71 tokens/s median with the packed kernel and 4.52 with the expanded BF16 kernel. The ranges overlapped across the two model loads per mode. The Radeon-only control reached 5.36 tokens/s on one request. During standalone verification, the native kernel matched the reference exactly on 454,656 synthetic scores spanning every finite FP16 scale encoding and all 128 bit positions.

Fresh installation requires Bionic Windows Vulkan llama.cpp 2.80.1, Phoenix XDNA 1 0.4.0 and a working XDNA 1 XRT driver. Installing the 0.4.0 prerequisite from scratch also requires Bionic CPU llama.cpp 2.80.1. Radeon modes require the AMD Vulkan driver. In the extracted archive run `.\install.ps1 -CheckOnly` and `.\install.ps1`, then select `phoenix-xdna1-win-x86_64-avx2@0.5.0` and reload the GGUF. The default placement uses CPU and XDNA. The package includes the C++ plugin, kernel source, device-program verifier, build instructions and checksums.

## Install

Install the Bionic Windows Vulkan llama.cpp 2.80.1 engine, Phoenix XDNA 1 0.4.0 and the XDNA 1 XRT driver. Radeon modes also require a registered AMD Vulkan driver and a single integrated AMD graphics adapter. The installer checks the inputs against runtime-lock.json and copies them into a separate versioned directory. It reads prerequisites from local installations. The package includes the new plugin, launcher and pair of XDNA device programs. It retains existing versions and uses your original GGUF.

Run the following in PowerShell 7 from the extracted pack.

```powershell
.\install.ps1 -CheckOnly
.\install.ps1
$runtime = Get-Content .\runtime-lock.json -Raw | ConvertFrom-Json
lms runtime select "$($runtime.name)@$($runtime.version)"
```

Alternatively select **Phoenix XDNA 1 (Windows)**, version **0.5.0**, under Bionic **Settings → Runtime → GGUF** and reload the model. Select **0.4.0** under the same entry to return to the previous release.

## Choose placement

| Mode | Model execution |
|---|---|
| `cpu-npu` (default) | CPU transformer computation, eligible XDNA prompt projections and a split CPU/XDNA vocabulary projection. The worker requests zero Radeon model layers. |
| `cpu-prompt` | Use eligible compact NPU prompt projections and CPU generation. |
| `cpu-prompt-resident` | Permit bounded resident BF16 storage of eligible NPU prompt weights. CPU handles generation. |
| `split-head` | Place model layers on the integrated Radeon. XDNA scores eligible output rows, and the CPU scores the remaining vocabulary rows. |
| `split-head-bf16` | Same hybrid placement with the previous expanded-BF16 head path. Use it to compare Q1 kernels in the same binary. |
| `prompt-ffn` | Place a supported first-block gate projection on XDNA for eligible prompt batches. Radeon handles the other supported weights. |
| `radeon` | Keep the XDNA plugin out of the request and use the integrated Radeon engine. |

```powershell
.\set-mode.ps1 -Mode split-head
```

Settings apply when Bionic next loads a model. The new path accepts contiguous Q1_0 GGUF rows at the compiled shape. It uploads exactly 2,211,840 packed weight bytes for that slice, plus its reusable activation and result buffers. The previous BF16 implementation uploads 31,457,280 weight bytes for the same slice. The remainder of Bonsai's output projection and transformer layers follow the selected placement. Other GGUF shapes use the pre-existing head or prompt kernels where eligible.

## Verification and rollback

For an instrumented request, create an empty file named `dispatch-test.log` in this version's installed backend directory before loading the model. A sampled line containing `storage=packed-q1` follows an actual completed packed-Q1 dispatch. The fallback line gives the cause if the packed program encounters a device or allocation error. The worker then uses the expanded-BF16 head path for subsequent eligible operations. Compare the trace offset before and after each request to distinguish chat work from model warm-up. The worker's startup and per-request logs include its process ID.

To return to the previous stable engine, unload the test instance, select `phoenix-xdna1-win-x86_64-avx2@0.4.0` and reload the model. The optional `remove.ps1` checks the selected engine and active processes, then archives only this installed version.

To rebuild the native Windows code, run `build.ps1 -XrtSdk <XRT SDK root>` with Visual Studio C++ Build Tools installed. Rebuilding a local development copy changes binary digests. Use `update-build-lock.ps1` only after verifying the intended binaries and programs. End-user installations use the supplied reviewed digest lock. The GGML headers and their license are under vendor/ggml.

To rebuild the device kernel, use the included `programs/source/build-programs.ps1` with `-Variant paired16 -Toolchain <IRON toolchain> -Output <output directory>`. The builder expects `python313`, `mlir-aie/ironenv` and `xrt_sdk/xrt` under that toolchain root. Files `bonsai_q1.py` and `bonsai_q1_vec.cc` implement the shipped paired design. The kernel consumes the original sign bits and FP16 block scales, BF16 activations, and returns FP32 scores. Nonfinite inputs or scales trigger the previous kernel path. The included scalar and eight-core variants reproduce the development controls.
