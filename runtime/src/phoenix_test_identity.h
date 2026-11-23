#pragma once
#include "phoenix_prefill.h"
#include <algorithm>
#include <cstring>
#include <vector>

// Each output row selects one input channel. Use real GGUF Q4_0 blocks
// for the compact program and BF16 rows for the small prompt program.
inline std::shared_ptr<const PhoenixPrefill::Prepared> phoenix_test_identity(
    PhoenixPrefill& engine, size_t rows, size_t columns) {
    if (rows == 2560) {
        return engine.prepare(rows, columns, [](void* arg, size_t row, uint16_t* dst) {
            const auto width = *static_cast<const size_t*>(arg);
            std::fill_n(dst, width, uint16_t{});
            dst[row % width] = 0x3f80;
        }, &columns);
    }
    const auto blocks = columns / 32;
    std::vector<uint8_t> raw(rows * blocks * 18, 0);
    for (size_t row = 0; row < rows; ++row) {
        const auto col = row % columns;
        auto* q = raw.data() + (row * blocks + col / 32) * 18;
        const ggml_fp16_t scale = 0x3c00;
        std::memcpy(q, &scale, 2);
        std::fill_n(q+2, 16, uint8_t{0x88});
        q[2+col%16] |= static_cast<uint8_t>(col % 32 < 16 ? 1 : 16);
    }
    return engine.prepare_q4(rows,columns,raw.data(),false);
}
