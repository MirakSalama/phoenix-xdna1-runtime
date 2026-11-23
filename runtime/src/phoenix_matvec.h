#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

// One lazily instantiated execution context per ggml backend. The caller supplies the
// selected GGUF model's decoded weights in unpadded, row-major BF16 format.
class PhoenixMatvec {
public:
    PhoenixMatvec();
    ~PhoenixMatvec();
    PhoenixMatvec(const PhoenixMatvec&) = delete;
    PhoenixMatvec& operator=(const PhoenixMatvec&) = delete;
    struct Prepared;
    using RowDecoder = void(*)(void* source, std::size_t row, std::uint16_t* output);
    static std::size_t packed_bytes(std::size_t rows, std::size_t columns);
    std::shared_ptr<const Prepared> prepare(std::size_t rows, std::size_t columns,
                                            RowDecoder decode, void* source);
    std::shared_ptr<const Prepared> prepare(const std::uint16_t* weights,
                                            std::size_t rows, std::size_t columns);
    void compute(const Prepared& weights, const float* vector, float* output);
    void compute(const std::uint16_t* weights, const float* vector, float* output,
                 std::size_t rows, std::size_t columns);

private:
    struct Impl;
    std::unique_ptr<Impl> context_;
};

// Native Bonsai Q1_0 output prefix. Each block holds an FP16 scale and 128 sign bits.
class PhoenixQ1Matvec {
public:
    static constexpr std::size_t rows = 3072;
    static constexpr std::size_t cols = 5120;
    static constexpr std::size_t row_bytes = cols / 128 * 18;
    static std::size_t packed_bytes(std::size_t height, std::size_t width);
    PhoenixQ1Matvec();
    ~PhoenixQ1Matvec();
    struct Prepared;
    std::shared_ptr<const Prepared> prepare(std::size_t height, std::size_t width,
                                            const std::uint8_t* packed_rows);
    void compute(const Prepared& weights, const float* vector, float* output);

private:
    struct Impl;
    std::unique_ptr<Impl> context_;
};
