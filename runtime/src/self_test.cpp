#include "phoenix_matvec.h"
#include "phoenix_prefill.h"
#include "phoenix_test_identity.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

static void test_prompt(std::size_t rows, std::size_t cols, std::size_t minimum) {
    constexpr std::size_t tokens = 128;
    if (PhoenixPrefill::supports(rows, cols, minimum - 1) ||
        !PhoenixPrefill::supports(rows, cols, minimum) || !PhoenixPrefill::supports(rows, cols, 512)) {
        throw std::runtime_error("Incorrect prompt batch bounds");
    }
    PhoenixPrefill prefill;
    auto weights = phoenix_test_identity(prefill, rows, cols);
    std::vector<float> input(tokens * cols), result(tokens * rows);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i % 11) + 1);
    prefill.compute(*weights, input.data(), result.data(), tokens);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t r = 0; r < rows; ++r) {
            if (result[t * rows + r] != input[t * cols + r % cols]) {
                throw std::runtime_error("Prompt hardware result differs from the known row permutation");
            }
        }
    }
}

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
        test_prompt(2560, 960, 64);
        test_prompt(7680, 2048, 128);
        test_prompt(6144, 1536, 128);
        test_prompt(12288, 1536, 128);
        test_prompt(1536, 6144, 128);
        test_prompt(2048, 7680, 128);
        std::puts("Phoenix XDNA 1 hardware self-test passed.");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Phoenix XDNA 1 self-test failed: %s\n", error.what());
        return 1;
    }
}
