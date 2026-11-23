// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// Any finite binary16 scale is exactly the sum of two BF16 values. Use integer
// bitfields so the core loop only needs native BF16 multipliers and FP32 MACs.
static inline void split_scale(uint16_t scale, bfloat16 &hi, bfloat16 &lo) {
    uint32_t exp = (scale >> 10) & 31u;
    uint32_t frac = scale & 1023u;
    uint32_t sign = scale & 0x8000u;
    if (exp == 0) {
        if (frac == 0) {
            hi = static_cast<bfloat16>(0.0f);
            lo = hi;
            return;
        }
        unsigned left = __builtin_clz(frac) - 21;
        exp = 113 - left;
        frac = (frac << left) & 1023u;
    } else {
        exp += 112;
    }
    const uint16_t lo_bits[8] = {0, 0, 128, 192, 256, 288, 320, 352};
    const unsigned rem = frac & 7;
    const uint16_t h = sign | (exp << 7) | (frac >> 3);
    const uint16_t l = rem ? sign | (((exp - 10) << 7) + lo_bits[rem]) : 0;
    hi = __builtin_bit_cast(bfloat16, h);
    lo = __builtin_bit_cast(bfloat16, l);
}

extern "C" void bonsai_score(const uint16_t *__restrict row,
                            uint16_t *__restrict x, float *__restrict scores,
                            int32_t mode, int32_t index) {
    if (mode != 1) {
        if (mode >= 2) {
            index = index * 2 + (mode - 2);
            if (index >= 15) index -= 15;
        }
        for (int i = 0; i < 360; i += 8)
            aie::store_v(x + index * 360 + i, aie::load_v<8>(row + i));
        return;
    }
    const auto negate = aie::broadcast<uint16_t, 32>(0x8000);
    const auto positive = aie::zeros<uint16_t, 32>();
    aie::accum<accfloat, 32> high = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> low = aie::zeros<accfloat, 32>();
    for (int b = 0; b < 40; ++b) {
        const uint16_t *block = row + b * 9;
        bfloat16 hi, lo;
        split_scale(block[0], hi, lo);
        for (int k = 0; k < 4; ++k) {
            const uint32_t mask = block[1 + 2*k] | uint32_t(block[2 + 2*k]) << 16;
            const auto sign = aie::select(negate, positive, aie::mask<32>::from_uint32(mask));
            const auto signed_x = aie::bit_xor(aie::load_v<32>(x + b*128 + k*32), sign).cast_to<bfloat16>();
            high = aie::mac(high, signed_x, hi);
            low = aie::mac(low, signed_x, lo);
        }
    }
    scores[index] = aie::reduce_add(aie::add(high.to_vector<float>(), low.to_vector<float>()));
}
