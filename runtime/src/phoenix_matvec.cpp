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
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t tile_rows = 65536;
constexpr std::size_t tile_cols = 1024;
constexpr std::size_t row_bytes = tile_cols * sizeof(std::uint16_t);
constexpr std::size_t heights[] = {
    128, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072,
    4096, 6144, 8192, 16384, 32768, 49152, tile_rows
};
std::size_t padded_height(std::size_t rows) {
    return *std::lower_bound(std::begin(heights), std::end(heights), rows);
}
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
    std::map<std::size_t, std::pair<xrt::bo, std::uint32_t>> instructions;
    std::size_t active_rows = 0;
    xrt::bo weight_bo, vector_bo, output_bo;
    xrt::run run;
    std::uint16_t* matrix = nullptr;
    std::uint16_t* vector;
    const float* result;

    Impl() {
        auto name = device.get_info<xrt::info::device::name>();
        if (name.find("Phoenix") == std::string::npos && name.find("Hawk") == std::string::npos) {
            throw std::runtime_error("Expected XDNA 1, found " + name);
        }
        const auto directory = program_directory();
        binary = xrt::xclbin(read_program(directory / "stream-gemv.xclbin"));
        device.register_xclbin(binary);
        hw = xrt::hw_context(device, binary.get_uuid());
        kernel = xrt::kernel(hw, "MLIR_AIE");
        vector_bo = xrt::bo(device, 2 * row_bytes, xrt::bo::flags::host_only, 0);
        output_bo = xrt::bo(device, tile_rows * sizeof(float), xrt::bo::flags::host_only, 0);
        vector = vector_bo.map<std::uint16_t*>();
        result = output_bo.map<float*>();
        std::fill_n(vector, tile_cols, std::uint16_t{0});
        run = xrt::run(kernel);
        run.set_arg(0, std::uint64_t{3});
        run.set_arg(4, vector_bo);
        run.set_arg(5, output_bo);
    }

    void select_rows(std::size_t rows) {
        if (active_rows == rows) return;
        auto it = instructions.find(rows);
        if (it == instructions.end()) {
            auto code = read_program(program_directory() / ("stream-" + std::to_string(rows) + ".bin"));
            if (code.size() % 4) throw std::runtime_error("Invalid GEMV instruction length");
            xrt::bo bo(device, code.size(), xrt::bo::flags::cacheable, kernel.group_id(1));
            bo.write(code.data(), code.size(), 0);
            bo.sync(to_device);
            it = instructions.emplace(rows, std::make_pair(std::move(bo), static_cast<std::uint32_t>(code.size()))).first;
        }
        run.set_arg(1, it->second.first);
        run.set_arg(2, it->second.second);
        vector[0] = static_cast<std::uint16_t>(rows / 128);
        vector_bo.sync(to_device, row_bytes, 0);
        active_rows = rows;
    }

    void select_scratch(std::size_t rows) {
        const auto bytes = rows * row_bytes;
        if (!matrix || weight_bo.size() < bytes) {
            weight_bo = xrt::bo(device, bytes, xrt::bo::flags::host_only, 0);
            matrix = weight_bo.map<std::uint16_t*>();
        }
        run.set_arg(3, weight_bo);
    }

    void execute() {
        run.start();
        auto state = run.wait(std::chrono::seconds(30));
        if (state == ERT_CMD_STATE_TIMEOUT) run.abort();
        if (state != ERT_CMD_STATE_COMPLETED) {
            throw std::runtime_error("Phoenix XRT GEMV failed, command state " + std::to_string(static_cast<int>(state)));
        }
    }

    void set_vector(const float* src, std::size_t width) {
        auto* x = vector + tile_cols;
        for (std::size_t i = 0; i < width; ++i) x[i] = to_bf16(src[i]);
        std::fill(x + width, x + tile_cols, static_cast<std::uint16_t>(0));
        vector_bo.sync(to_device, row_bytes, row_bytes);
    }

    void read_result(float* dst, std::size_t count, bool add) {
        output_bo.sync(from_device, count * sizeof(float), 0);
        if (!add) std::copy_n(result, count, dst);
        else for (std::size_t i = 0; i < count; ++i) dst[i] += result[i];
    }
};

PhoenixMatvec::PhoenixMatvec() : context_(std::make_unique<Impl>()) {}
PhoenixMatvec::~PhoenixMatvec() = default;

struct PhoenixMatvec::Prepared {
    std::size_t rows, columns;
    std::vector<xrt::bo> tiles;
};

std::size_t PhoenixMatvec::packed_bytes(std::size_t rows, std::size_t columns) {
    if (!rows || !columns) throw std::invalid_argument("Empty Phoenix matrix");
    const auto across = 1 + (columns - 1) / tile_cols;
    auto rounded_rows = rows - 1 - ((rows - 1) % tile_rows);
    auto tail = padded_height(1 + ((rows - 1) % tile_rows));
    const auto limit = std::numeric_limits<std::size_t>::max() / across / row_bytes;
    if (rounded_rows > limit || tail > limit - rounded_rows) {
        throw std::length_error("Phoenix matrix too large");
    }
    return (rounded_rows + tail) * across * row_bytes;
}

std::shared_ptr<const PhoenixMatvec::Prepared> PhoenixMatvec::prepare(
    std::size_t rows, std::size_t columns, RowDecoder decode, void* source) {
    if (!decode) throw std::invalid_argument("Missing Phoenix weight decoder");
    packed_bytes(rows, columns);
    auto dst = std::make_shared<Prepared>();
    dst->rows = rows;
    dst->columns = columns;
    const auto down = 1 + (rows - 1) / tile_rows;
    const auto across = 1 + (columns - 1) / tile_cols;
    const auto tail = padded_height(1 + (rows - 1) % tile_rows);
    dst->tiles.reserve(down * across);
    std::vector<std::uint16_t*> mapped;
    for (std::size_t t = 0; t < down * across; ++t) {
        const auto height = t % down == down - 1 ? tail : tile_rows;
        auto& bo = dst->tiles.emplace_back(context_->device, height * row_bytes, xrt::bo::flags::host_only, 0);
        mapped.push_back(bo.map<std::uint16_t*>());
    }
    std::vector<std::uint16_t> scratch(across > 1 ? columns : 0);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto row_tile = row / tile_rows, in_tile = row % tile_rows * tile_cols;
        if (across == 1) {
            auto* out = mapped[row_tile] + in_tile;
            decode(source, row, out);
            std::fill(out + columns, out + tile_cols, std::uint16_t{0});
        } else {
            decode(source, row, scratch.data());
            for (std::size_t col = 0; col < columns; col += tile_cols) {
                auto* out = mapped[(col / tile_cols) * down + row_tile] + in_tile;
                const auto width = std::min(tile_cols, columns - col);
                std::copy_n(scratch.data() + col, width, out);
                std::fill(out + width, out + tile_cols, std::uint16_t{0});
            }
        }
    }
    if (const auto remainder = rows % tile_rows) {
        for (std::size_t t = down - 1; t < mapped.size(); t += down) {
            std::fill(mapped[t] + remainder * tile_cols, mapped[t] + tail * tile_cols, std::uint16_t{0});
        }
    }
    for (auto& bo : dst->tiles) bo.sync(to_device);
    return dst;
}

std::shared_ptr<const PhoenixMatvec::Prepared> PhoenixMatvec::prepare(
    const std::uint16_t* weights, std::size_t rows, std::size_t columns) {
    struct Source { const std::uint16_t* data; std::size_t stride; } source{weights, columns};
    if (!weights) throw std::invalid_argument("Missing Phoenix matrix");
    return prepare(rows, columns, [](void* opaque, std::size_t row, std::uint16_t* out) {
        const auto& s = *static_cast<const Source*>(opaque);
        std::copy_n(s.data + row * s.stride, s.stride, out);
    }, &source);
}

void PhoenixMatvec::compute(const Prepared& weights, const float* vector, float* output) {
    if (!vector || !output) throw std::invalid_argument("Invalid Phoenix GEMV inputs");
    auto& p = *context_;
    std::size_t tile = 0;
    for (std::size_t col = 0; col < weights.columns; col += tile_cols) {
        p.set_vector(vector + col, std::min(tile_cols, weights.columns - col));
        for (std::size_t row = 0; row < weights.rows; row += tile_rows) {
            p.select_rows(padded_height(std::min(tile_rows, weights.rows - row)));
            p.run.set_arg(3, weights.tiles[tile++]);
            p.execute();
            p.read_result(output + row, std::min(tile_rows, weights.rows - row), col != 0);
        }
    }
}

void PhoenixMatvec::compute(const std::uint16_t* weights, const float* vector, float* output,
                            std::size_t rows, std::size_t columns) {
    if (!weights || !vector || !output || rows == 0 || columns == 0 ||
        rows > std::numeric_limits<std::size_t>::max() / columns / sizeof(*weights)) {
        throw std::invalid_argument("Invalid Phoenix GEMV dimensions or input");
    }
    auto& p = *context_;
    for (std::size_t col = 0; col < columns; col += tile_cols) {
        const auto width = std::min(tile_cols, columns - col);
        p.set_vector(vector + col, width);
        for (std::size_t row = 0; row < rows; row += tile_rows) {
            const auto height = std::min(tile_rows, rows - row);
            const auto padded = padded_height(height);
            p.select_rows(padded);
            p.select_scratch(padded);
            if (columns == tile_cols) {
                std::memcpy(p.matrix, weights + row * columns, height * tile_cols * sizeof(*weights));
            } else {
                for (std::size_t j = 0; j < height; ++j) {
                    auto* dest = p.matrix + j * tile_cols;
                    std::memcpy(dest, weights + (row + j) * columns + col, width * sizeof(*weights));
                    std::fill(dest + width, dest + tile_cols, static_cast<std::uint16_t>(0));
                }
            }
            std::fill_n(p.matrix + height * tile_cols, (padded - height) * tile_cols, std::uint16_t{0});
            p.weight_bo.sync(to_device, padded * row_bytes, 0);
            p.execute();
            p.read_result(output + row, height, col != 0);
        }
    }
}
