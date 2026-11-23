// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void phoenix_stream(const bfloat16 *__restrict weights,
                               bfloat16 *__restrict activation,
                               float *__restrict output, int32_t *groups,
                               int32_t mode, int32_t row) {
    if (mode == 0) {
        groups[0] = reinterpret_cast<const uint16_t *>(weights)[0];
        return;
    }
    if (mode == 1) {
        for (int i = 0; i < 1024; i += 32)
            aie::store_v(activation + i, aie::load_v<32>(weights + i));
        return;
    }
    aie::accum<accfloat, 16> a = aie::zeros<accfloat, 16>();
    aie::accum<accfloat, 16> b = aie::zeros<accfloat, 16>();
    for (int i = 0; i < 1024; i += 32) {
        a = aie::mac(a, aie::load_v<16>(weights + i), aie::load_v<16>(activation + i));
        b = aie::mac(b, aie::load_v<16>(weights + i + 16), aie::load_v<16>(activation + i + 16));
    }
    aie::vector<float, 16> sum = aie::add(a.to_vector<float>(), b.to_vector<float>());
    output[row] = aie::reduce_add(sum);
}
