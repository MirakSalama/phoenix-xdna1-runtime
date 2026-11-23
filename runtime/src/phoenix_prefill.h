#pragma once

#include "phoenix_matvec.h"

// Projects contiguous token-major F32 activations using resident row-major BF16 weights.
class PhoenixPrefill {
public:
    PhoenixPrefill();
    ~PhoenixPrefill();
    PhoenixPrefill(const PhoenixPrefill&) = delete;
    PhoenixPrefill& operator=(const PhoenixPrefill&) = delete;
    struct Prepared;
    static bool supports(std::size_t rows, std::size_t columns, std::size_t tokens);
    static std::size_t packed_bytes(std::size_t rows, std::size_t columns);
    // Enable concurrent row decoding only when the supplied decoder supports it.
    void set_prepare_threads(int threads);
    std::shared_ptr<const Prepared> prepare(std::size_t rows, std::size_t columns,
                                            PhoenixMatvec::RowDecoder decode, void* source);
    void compute(const Prepared& weights, const float* activations, float* output, std::size_t tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> context_;
    int prepare_threads_ = 1;
};
