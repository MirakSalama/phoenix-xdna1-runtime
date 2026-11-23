#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

// Q4_0 stores 32 signed four-bit values and one binary16 scale. Each
// 64x64 device tile contains 32 records of four rows and 32 columns.
// Subdivide each record into four 4x8 matrices in the AIE MMUL lane order.
inline void phoenix_pack_q4_prompt_tiles(const std::uint8_t* weights, bool repacked,
                                        std::size_t columns, std::size_t first_tile,
                                        std::size_t past_tile, std::uint8_t* output) {
    const std::size_t blocks = columns / 32;
    for (std::size_t tile = first_tile; tile < past_tile; ++tile) {
        auto* record = output + tile * 64 * blocks * 20;
        for (std::size_t col = 0; col < columns; col += 64) {
            for (std::size_t group = 0; group < 64; group += 4) {
                for (std::size_t half = 0; half < 2; ++half, record += 80) {
                    const std::size_t block = col / 32 + half;
                    for (std::size_t lane = 0; lane < 4; ++lane) {
                        const std::size_t row = tile * 64 + group + lane;
                        const std::uint8_t* scale_ptr;
                        const std::uint8_t* codes;
                        std::size_t code_stride;
                        if (repacked) {
                            // The pinned AVX2 CPU_REPACK implementation stores eight
                            // rows together, with two eight-byte runs per row.
                            const auto* src = weights + (row / 8 * blocks + block) * 144;
                            scale_ptr = src + (row % 8) * 2;
                            codes = src + 16 + (row % 8) * 8;
                            code_stride = 64;
                        } else {
                            scale_ptr = weights + (row * blocks + block) * 18;
                            codes = scale_ptr + 2;
                            code_stride = 8;
                        }
                        ggml_fp16_t d;
                        std::memcpy(&d, scale_ptr, sizeof(d));
                        const float scale = ggml_fp16_to_fp32(d);
                        const auto hi = ggml_fp32_to_bf16(scale);
                        const auto lo = (d & 0x7c00) == 0x7c00 ? ggml_bf16_t{} :
                            ggml_fp32_to_bf16(scale == 0.0f ? scale : scale - ggml_bf16_to_fp32(hi));
                        std::memcpy(record + 64 + 2 * lane, &hi, 2);
                        std::memcpy(record + 72 + 2 * lane, &lo, 2);
                        for (std::size_t part = 0; part < 4; ++part) {
                            const auto* q = codes + (part % 2) * code_stride;
                            auto* target = record + 16 * part + 4 * lane;
                            const unsigned shift = part >= 2 ? 4 : 0;
                            for (std::size_t j = 0; j < 4; ++j) {
                                const auto pair = ((q[2*j] >> shift) & 15) | (((q[2*j+1] >> shift) & 15) << 4);
                                target[j] = static_cast<std::uint8_t>(pair ^ (repacked ? 0 : 0x88));
                            }
                        }
                    }
                }
            }
        }
    }
}
