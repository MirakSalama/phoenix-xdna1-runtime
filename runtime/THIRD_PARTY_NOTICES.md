# Third-party notices

The GGML headers under `vendor/ggml` come from llama.cpp. They retain their MIT license in `vendor/ggml/LICENSE`. The pinned GGML revision for this release is 29b52d398.

Phoenix AI's initial first-generation XDNA chat work grew from Abhiram Jayasankar’s Ryzen AI NPU Lab. Its MIT notice is included in `licenses/ryzen-ai-npu-lab-MIT.txt`. [Upstream experiment archive](https://github.com/AbhiramJayasankar/ryzenai-npu-lab/tree/a9531b87b770fd3142164b61254fb6f74fed2d22)

The device program sources identify their licenses in their file headers. Device programs adapted from AMD MLIR-AIE use Apache License 2.0 with the LLVM exception. Copies of both texts accompany the programs in `programs/Apache-2.0.txt` and `programs/LLVM-exception.txt`. Bundled whole-array source, where present, retains its upstream notices and license in `programs/source/amd-whole-array/LICENSE`. Phoenix parameterizes those kernels for its own operation shapes and packing layouts.

The device kernels use AMD’s header-only AIE API. Its MIT notice is included in `licenses/aie-api-MIT.txt`. [AMD AIE API](https://github.com/Xilinx/aie_api)

The Windows host links to AMD XRT userspace interfaces under Apache License 2.0. XRT headers used to build the host carry copyrights of Xilinx, Inc. and Advanced Micro Devices, Inc. The SDK's upstream license text is included in `licenses/xrt-sdk-LICENSE.txt`. XRT and its driver are provided by the user's installation. Bionic's native engine and bindings are also obtained from the user's installation. The archive contains the Phoenix host binaries and device programs built for this version.

Copyright and license markings apply to the components that carry them. This package does not grant a separate project-wide license for otherwise unlicensed material.
