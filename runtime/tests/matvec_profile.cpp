#include "../src/phoenix_matvec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

template<class F> static double median_ms(F call, unsigned repeats) {
    call();
    std::vector<double> samples;
    for (unsigned i = 0; i < repeats; ++i) {
        const auto begin = Clock::now();
        call();
        samples.push_back(elapsed_ms(begin));
    }
    std::sort(samples.begin(), samples.end());
    return samples[repeats / 2];
}

static void profile(PhoenixMatvec& npu, size_t rows, size_t cols) {
    std::vector<std::uint16_t> weights(rows * cols);
    std::vector<float> input(cols), output(rows), expected(rows);
    for (size_t c = 0; c < cols; ++c) input[c] = float(int(c % 19) - 9) / 16;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const float w = float(int((r * 11 + c * 7 + r * (c % 5)) % 29) - 14) / 16;
            std::uint32_t bits;
            std::memcpy(&bits, &w, sizeof(bits));
            weights[r * cols + c] = static_cast<std::uint16_t>(bits >> 16);
            expected[r] += input[c] * w;
        }
    }
    const auto start = Clock::now();
    auto prepared = npu.prepare(weights.data(), rows, cols);
    const auto prepare_ms = elapsed_ms(start);
    const unsigned repeats = rows > 10000 ? 9 : 41;
    const auto verify = [&]() {
        if (output != expected) throw std::runtime_error("CPU reference mismatch");
    };
    const double cached = median_ms([&]() { npu.compute(*prepared, input.data(), output.data()); }, repeats);
    verify();
    const double raw = median_ms([&]() {
        npu.compute(weights.data(), input.data(), output.data(), rows, cols);
    }, repeats);
    verify();
    std::printf("%zu,%zu,%zu,%.3f,%.3f,%.3f\n", rows, cols,
                npu.packed_bytes(rows, cols), prepare_ms, cached, raw);
    std::fflush(stdout);
}

int main() {
    try {
        PhoenixMatvec npu;
        std::puts("rows,columns,resident_bytes,prepare_ms,warm_ms,uncached_ms");
        for (const auto& shape : {std::pair<size_t, size_t>{3072,1024}, {2560,960},
                {512,1024}, {1024,1024}, {2048,1024}, {6144,1024}, {320,960},
                {960,2560}, {3087,1037}, {49152,960}, {65536,1024}, {65539,1057}})
            profile(npu, shape.first, shape.second);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
