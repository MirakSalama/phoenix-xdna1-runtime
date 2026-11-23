#pragma once

#include "phoenix_matvec.h"
#include "ggml.h"

// Projects contiguous token-major F32 activations using resident NPU weights.
class PhoenixPrefill {
public:
    PhoenixPrefill();
    ~PhoenixPrefill();
    PhoenixPrefill(const PhoenixPrefill&) = delete;
    PhoenixPrefill& operator=(const PhoenixPrefill&) = delete;
    struct Prepared;
    static bool supports(std::size_t rows, std::size_t columns, std::size_t tokens);
    static std::size_t packed_bytes(std::size_t rows, std::size_t columns);
    static void decode_q5_0_row(const void* packed, std::uint16_t* dst, std::size_t columns);
    // Enable concurrent row decoding only when the supplied decoder supports it.
    void set_prepare_threads(int threads);
    std::shared_ptr<const Prepared> prepare(std::size_t rows, std::size_t columns,
                                            PhoenixMatvec::RowDecoder decode, void* source);
    std::shared_ptr<const Prepared> prepare_q4(std::size_t rows, std::size_t columns,
                                               const void* data, bool cpu_repacked);
    // Returns true when the previous byte-identical activation set was reused.
    bool compute(const Prepared& weights, const float* activations, float* output, std::size_t tokens,
                 ggml_type weight_type = GGML_TYPE_F32);

private:
    struct Impl;
    std::unique_ptr<Impl> context_;
    int prepare_threads_ = 1;
};
