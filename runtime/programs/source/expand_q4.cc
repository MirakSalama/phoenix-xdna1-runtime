// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// A 64x64 tile has 32 compact records of four rows by 32 columns.
// Each record contains four consecutive signed-int4 4x8 microtiles,
// then four high and four low block scales. Binary16 scales from Q4_0
// are exactly representable as the sum of these two BF16 components.
extern "C" void phoenix_expand_q4(const uint8_t *__restrict in,
                                  bfloat16 *__restrict out) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int g = 0; g < 32; ++g) {
        const uint8_t *record = in + g*80;
        const auto *scale = reinterpret_cast<const bfloat16 *>(record + 64);
        aie::vector<bfloat16, 32> hi, lo;
        for (int row = 0; row < 4; ++row) {
            hi.insert(row,aie::broadcast<bfloat16,8>(scale[row]));
            lo.insert(row,aie::broadcast<bfloat16,8>(scale[4+row]));
        }
        for (int j = 0; j < 4; ++j) {
            auto packed = aie::load_v<32>(reinterpret_cast<const int4 *>(record + 16*j));
            auto converted = aie::to_float<bfloat16>(aie::unpack(aie::unpack(packed)));
            auto values = aie::mac(aie::mul(converted,hi),converted,lo);
            aie::store_v(out+g*128+j*32,values.to_vector<bfloat16>());
        }
    }
}
