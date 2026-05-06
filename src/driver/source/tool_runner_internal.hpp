#ifndef TOOL_RUNNER_INTERNAL_HPP_
#define TOOL_RUNNER_INTERNAL_HPP_

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "driver/driver_options.hpp"

namespace tc::driver {

struct LoweredPaths {
    std::filesystem::path temp_dir;
    std::filesystem::path input_mlir;
    std::filesystem::path lowered_mlir;
    std::filesystem::path llvm_ir;
};

void RunCommand(const std::vector<std::string>& argv);
void RemoveTree(const std::filesystem::path& path);
LoweredPaths LowerMlirToLlvm(const std::string& mlir_text);
std::vector<std::string> BuildLlcCommand(const DriverOptions& opt,
                                         const std::filesystem::path& input,
                                         std::string_view filetype,
                                         const std::filesystem::path& output);

} // namespace tc::driver

#endif // TOOL_RUNNER_INTERNAL_HPP_
