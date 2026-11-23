// Validate the production row unpacker and XRT activation cache against stock GGML rounding.
#include "phoenix_prefill.h"
#include "phoenix_test_identity.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

void verify_unpack() {
    // Every binary16 scale, including both signed zeros and all NaNs, with independently chosen codes.
    std::vector<block_q5_0> codes(65536);
    std::mt19937 rng(811);
    for (size_t i=0; i<codes.size(); ++i) {
        auto& q = codes[i];
        q.d = static_cast<uint16_t>(i);
        for (auto& b : q.qh) b = static_cast<uint8_t>(rng());
        for (auto& b : q.qs) b = static_cast<uint8_t>(rng());
    }
    const auto n = codes.size()*QK5_0;
    std::vector<float> f(n);
    std::vector<ggml_bf16_t> expected(n);
    std::vector<uint16_t> actual(n);
    ggml_get_type_traits(GGML_TYPE_Q5_0)->to_float(codes.data(),f.data(),n);
    ggml_fp32_to_bf16_row(f.data(),expected.data(),n);
    PhoenixPrefill::decode_q5_0_row(codes.data(),actual.data(),n);
    require(!std::memcmp(actual.data(),expected.data(),n*sizeof(uint16_t)), "Q5_0 unpack changed BF16 words");
}

void project(PhoenixPrefill& engine, const PhoenixPrefill::Prepared& w, size_t rows, size_t cols,
             size_t tokens, const std::vector<float>& input, ggml_type weight_type, bool reuse) {
    std::vector<float> expected(input);
    const auto dot = ggml_get_type_traits_cpu(weight_type)->vec_dot_type;
    if (dot == GGML_TYPE_Q8_0) {
        std::vector<uint8_t> codes(ggml_row_size(dot,input.size()));
        ggml_get_type_traits_cpu(dot)->from_float(input.data(),codes.data(),input.size());
        ggml_get_type_traits(dot)->to_float(codes.data(),expected.data(),input.size());
    }
    std::vector<ggml_bf16_t> bf(input.size());
    ggml_fp32_to_bf16_row(expected.data(),bf.data(),expected.size());
    ggml_bf16_to_fp32_row(bf.data(),expected.data(),expected.size());
    std::vector<float> out(tokens*rows);
    require(engine.compute(w,input.data(),out.data(),tokens,weight_type) == reuse, "incorrect activation cache decision");
    for (size_t t=0;t<tokens;++t) for (size_t r=0;r<rows;++r) {
        require(out[t*rows+r] == expected[t*cols+r%cols], "hardware output differs from stock-rounded reference");
    }
}
}

int main() {
    try {
        verify_unpack();
        PhoenixPrefill engine;
        engine.set_prepare_threads(4);
        struct Shape { size_t rows, cols; std::shared_ptr<const PhoenixPrefill::Prepared> w; };
        std::vector<Shape> shapes{{2560,960},{6144,1536},{7680,2048},{12288,1536},{1536,6144},{2048,7680}};
        for(auto& s : shapes) s.w = phoenix_test_identity(engine,s.rows,s.cols);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> sample(-1.f,1.f);
        for (size_t n : {128,257,511}) {
            for (const auto& s : shapes) {
                std::vector<float> input(n*s.cols);
                for (auto& v : input) v = sample(rng);
                auto same_bytes = input;
                project(engine,*s.w,s.rows,s.cols,n,input,GGML_TYPE_Q5_0,false);
                project(engine,*s.w,s.rows,s.cols,n,same_bytes,GGML_TYPE_Q5_0,true);
                input.back() = 3.f;
                project(engine,*s.w,s.rows,s.cols,n,input,GGML_TYPE_Q5_0,false);
                project(engine,*s.w,s.rows,s.cols,n,input,GGML_TYPE_F32,false);
                std::printf("%zux%zu batch %zu: distinct pointer hit, in-place mutation, dot rounding and every output element verified\n",
                            s.rows,s.cols,n);
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr,"%s\n",error.what());
        return 1;
    }
}
