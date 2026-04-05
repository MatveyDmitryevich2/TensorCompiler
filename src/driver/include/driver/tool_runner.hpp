#ifndef TOOL_RUNNER_HPP_
#define TOOL_RUNNER_HPP_

#include <filesystem>
#include <string>

#include "driver/driver_options.hpp"

namespace tc::driver {

void SetupLogging(int argc, const char* argv[]);
std::string ReadTextFile(const std::filesystem::path& path);
void WriteTextFile(const std::string& path, const std::string& text);
void LowerToLlvmAndAsm(const DriverOptions& opt, const std::string& mlir_text);

} // namespace tc::driver

#endif // TOOL_RUNNER_HPP_
