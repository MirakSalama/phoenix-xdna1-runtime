#include "phoenix_matvec.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t tile_rows = 3072;
constexpr std::size_t tile_cols = 1024;
constexpr auto to_device = XCL_BO_SYNC_BO_TO_DEVICE;
constexpr auto from_device = XCL_BO_SYNC_BO_FROM_DEVICE;

std::filesystem::path program_directory() {
    // An address inside this DLL identifies the pack even when Bionic changes cwd.
    static const wchar_t module_anchor = 0;
    HMODULE module{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, &module_anchor, &module)) {
        throw std::runtime_error("Could not locate the Phoenix module");
    }
    std::wstring filename(32768, 0);
    const auto count = GetModuleFileNameW(module, filename.data(), static_cast<DWORD>(filename.size()));
    if (!count || count >= filename.size()) throw std::runtime_error("Could not locate the Phoenix NPU program files");
    filename.resize(count);
    return std::filesystem::path(filename).parent_path() / "programs";
}

std::vector<char> read_program(const std::filesystem::path& filename) {
    std::ifstream input(filename, std::ios::binary | std::ios::ate);
    auto bytes = input.tellg();
    if (!input || bytes <= 0 || bytes > (1 << 24)) {
        throw std::runtime_error("Missing or invalid Phoenix program: " + filename.string());
    }
    std::vector<char> contents(static_cast<std::size_t>(bytes));
    input.seekg(0);
    if (!input.read(contents.data(), bytes)) throw std::runtime_error("Could not read " + filename.string());
    return contents;
}

std::uint16_t to_bf16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040U);
    }
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
}

struct PhoenixMatvec::Impl {
    xrt::device device{0};
    xrt::xclbin binary;
    xrt::hw_context hw;
    xrt::kernel kernel;
    xrt::bo instructions, weight_bo, vector_bo, output_bo;
    xrt::run run;
    std::uint16_t* matrix;
    std::uint16_t* vector;
    const float* result;

    Impl() {
        auto name = device.get_info<xrt::info::device::name>();
        if (name.find("Phoenix") == std::string::npos && name.find("Hawk") == std::string::npos) {
            throw std::runtime_error("Expected XDNA 1, found " + name);
        }
        const auto directory = program_directory();
        binary = xrt::xclbin(read_program(directory / "gemv-3072x1024.xclbin"));
        auto code = read_program(directory / "gemv-3072x1024.bin");
        if (code.size() % 4 != 0) throw std::runtime_error("Invalid GEMV instruction length");
        device.register_xclbin(binary);
        hw = xrt::hw_context(device, binary.get_uuid());
        kernel = xrt::kernel(hw, "MLIR_AIE");
        instructions = xrt::bo(device, code.size(), xrt::bo::flags::cacheable, kernel.group_id(1));
        instructions.write(code.data(), code.size(), 0);
        instructions.sync(to_device);
        weight_bo = xrt::bo(device, tile_rows * tile_cols * sizeof(std::uint16_t), xrt::bo::flags::host_only, 0);
        vector_bo = xrt::bo(device, tile_cols * sizeof(std::uint16_t), xrt::bo::flags::host_only, 0);
        output_bo = xrt::bo(device, tile_rows * sizeof(float), xrt::bo::flags::host_only, 0);
        matrix = weight_bo.map<std::uint16_t*>();
        vector = vector_bo.map<std::uint16_t*>();
        result = output_bo.map<float*>();
        std::fill_n(matrix, tile_rows * tile_cols, static_cast<std::uint16_t>(0));
        weight_bo.sync(to_device);
        run = xrt::run(kernel);
        run.set_arg(0, std::uint64_t{3});
        run.set_arg(1, instructions);
        run.set_arg(2, static_cast<std::uint32_t>(code.size()));
        run.set_arg(3, weight_bo);
        run.set_arg(4, vector_bo);
        run.set_arg(5, output_bo);
    }

    void execute() {
        run.start();
        auto state = run.wait(std::chrono::seconds(30));
        if (state == ERT_CMD_STATE_TIMEOUT) run.abort();
        if (state != ERT_CMD_STATE_COMPLETED) {
            throw std::runtime_error("Phoenix XRT GEMV failed, command state " + std::to_string(static_cast<int>(state)));
        }
    }
};

PhoenixMatvec::PhoenixMatvec() : context_(std::make_unique<Impl>()) {}
PhoenixMatvec::~PhoenixMatvec() = default;

void PhoenixMatvec::compute(const std::uint16_t* weights, const float* vector, float* output,
                            std::size_t rows, std::size_t columns) {
    if (!weights || !vector || !output || rows == 0 || columns == 0 ||
        rows > std::numeric_limits<std::size_t>::max() / columns / sizeof(*weights)) {
        throw std::invalid_argument("Invalid Phoenix GEMV dimensions or input");
    }
    auto& p = *context_;
    for (std::size_t col = 0; col < columns; col += tile_cols) {
        const auto width = std::min(tile_cols, columns - col);
        for (std::size_t i = 0; i < width; ++i) p.vector[i] = to_bf16(vector[col + i]);
        std::fill(p.vector + width, p.vector + tile_cols, static_cast<std::uint16_t>(0));
        p.vector_bo.sync(to_device);
        for (std::size_t row = 0; row < rows; row += tile_rows) {
            const auto height = std::min(tile_rows, rows - row);
            if (columns == tile_cols) {
                std::memcpy(p.matrix, weights + row * columns, height * tile_cols * sizeof(*weights));
            } else {
                for (std::size_t j = 0; j < height; ++j) {
                    auto* dest = p.matrix + j * tile_cols;
                    std::memcpy(dest, weights + (row + j) * columns + col, width * sizeof(*weights));
                    std::fill(dest + width, dest + tile_cols, static_cast<std::uint16_t>(0));
                }
            }
            p.weight_bo.sync(to_device, height * tile_cols * sizeof(*weights), 0);
            p.execute();
            p.output_bo.sync(from_device, height * sizeof(float), 0);
            if (col == 0) std::copy_n(p.result, height, output + row);
            else for (std::size_t j = 0; j < height; ++j) output[row + j] += p.result[j];
        }
    }
}
