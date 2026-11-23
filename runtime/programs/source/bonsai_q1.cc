// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// Q1_0 stores one IEEE binary16 scale and 128 sign bits in 18 bytes.
static inline float scale_f32(uint16_t v) {
    uint32_t exponent = (v >> 10) & 31u;
    uint32_t mantissa = v & 1023u;
    uint32_t sign = uint32_t(v & 0x8000u) << 16;
    uint32_t bits;
    if (exponent == 0) {
        if (!mantissa) return 0.0f;
        unsigned shift = __builtin_clz(mantissa) - 21;
        bits = sign | ((113u - shift) << 23) | ((mantissa << shift & 1023u) << 13);
    } else {
        bits = sign | ((exponent == 31 ? 255u : exponent + 112u) << 23) | (mantissa << 13);
    }
    return __builtin_bit_cast(float, bits);
}

extern "C" void bonsai_score(const uint16_t *__restrict row,
                            uint16_t *__restrict x_bits, float *__restrict scores,
                            int32_t mode, int32_t index) {
    if (mode == 0) {
        for (int i = 0; i < 360; i += 8)
            aie::store_v(x_bits + index * 360 + i, aie::load_v<8>(row + i));
        return;
    }
    const auto *x = reinterpret_cast<const bfloat16 *>(x_bits);
    const auto minus = aie::broadcast<bfloat16, 32>(static_cast<bfloat16>(-1.0f));
    const auto plus = aie::broadcast<bfloat16, 32>(static_cast<bfloat16>(1.0f));
    float score = 0.0f;
    for (int b = 0; b < 40; ++b) {
        const uint16_t *block = row + b * 9;
        aie::accum<accfloat, 32> sum = aie::zeros<accfloat, 32>();
        for (int k = 0; k < 4; ++k) {
            const uint32_t mask = block[1 + 2*k] | uint32_t(block[2 + 2*k]) << 16;
            auto signs = aie::select(minus, plus, aie::mask<32>::from_uint32(mask));
            sum = aie::mac(sum, aie::load_v<32>(x + b * 128 + k * 32), signs);
        }
        score += scale_f32(block[0]) * aie::reduce_add(sum.to_vector<float>());
    }
    scores[index] = score;
}
