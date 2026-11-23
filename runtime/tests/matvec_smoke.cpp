#include "../src/phoenix_matvec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

int main() {
    try {
        PhoenixMatvec npu;
        for (int side = 0; side < 2; ++side) {
            const int rows = side ? 3087 : 3072, cols = side ? 1037 : 1024;
            std::vector<std::uint16_t> weights(rows * cols);
            std::vector<float> vec(cols), expected(rows), actual(rows);
            // BF16-exact operands and integer-exact partial sums. Test output ignores
            // no nonzero rows, including the bottom and right edges of partial tiles.
            for (int c = 0; c < cols; ++c) vec[c] = static_cast<float>((3 * c + 1) % 19 - 9) / 16;
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    float w = static_cast<float>((r * 11 + c * 7 + r * (c % 5)) % 29 - 14) / 16;
                    std::uint32_t bits{};
                    std::memcpy(&bits, &w, sizeof(bits));
                    weights[r * cols + c] = static_cast<std::uint16_t>(bits >> 16);
                    expected[r] += w * vec[c];
                }
            }
            auto start = std::chrono::steady_clock::now();
            npu.compute(weights.data(), vec.data(), actual.data(), rows, cols);
            auto end = std::chrono::steady_clock::now();
            double max_error = 0;
            for (int r = 0; r < rows; ++r) {
                if (!std::isfinite(actual[r])) return 1;
                max_error = std::max(max_error, std::abs(double(actual[r]) - expected[r]));
            }
            std::printf("%dx%d max_abs_error %.6f, wall %.2f ms\n", rows, cols, max_error,
                        std::chrono::duration<double, std::milli>(end - start).count());
            if (max_error != 0.0) return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
