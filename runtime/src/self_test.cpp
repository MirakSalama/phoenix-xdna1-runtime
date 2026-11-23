#include "phoenix_matvec.h"
#include "phoenix_prefill.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <vector>

// Called by the staged launcher so installation verifies its own DLLs, program
// files, XRT device and a numerical result before making the pack selectable.
extern "C" __declspec(dllexport) int phoenix_xdna_self_test() noexcept {
    try {
        PhoenixMatvec npu;
        const std::uint16_t weights[] = {0x3f80, 0x4000, 0xbf80, 0x4000, 0xbf80, 0x3f80};
        const float x[] = {3, 2, 1};
        float y[2]{};
        auto prepared = npu.prepare(weights, 2, 3);
        npu.compute(*prepared, x, y);
        if (y[0] != 6 || y[1] != 5) {
            std::fprintf(stderr, "Phoenix XDNA 1 self-test failed. Expected [6, 5], received [%g, %g].\n", y[0], y[1]);
            return 1;
        }
        static constexpr size_t rows = 2560, cols = 960, tokens = 128;
        PhoenixPrefill prefill;
        auto batch_weights = prefill.prepare(rows, cols, [](void *, size_t row, std::uint16_t * dest) {
            std::fill_n(dest, cols, std::uint16_t{0});
            dest[row % cols] = 0x3f80;
        }, nullptr);
        std::vector<float> input(tokens * cols), result(tokens * rows);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 11) + 1);
        prefill.compute(*batch_weights, input.data(), result.data(), tokens);
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t r = 0; r < rows; ++r) {
                if (result[t * rows + r] != input[t * cols + r % cols]) {
                    std::fprintf(stderr, "Phoenix prefill self-test failed at token %zu, row %zu.\n", t, r);
                    return 1;
                }
            }
        }
        std::puts("Phoenix XDNA 1 hardware self-test passed.");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Phoenix XDNA 1 self-test failed: %s\n", error.what());
        return 1;
    }
}
