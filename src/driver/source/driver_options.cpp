#include "driver/driver_options.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tc::driver {

namespace {

std::string RequireValue(int argc, const char* argv[], int& i, std::string_view flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error{"missing value for flag " + std::string(flag)};
    }
    ++i;
    return argv[i];
}

std::pair<std::string, std::string> ParseInputSpec(const std::string& spec) {
    const size_t eq = spec.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) {
        throw std::runtime_error{"--input expects name=path"};
    }
    return {spec.substr(0, eq), spec.substr(eq + 1)};
}

} // namespace

std::string Usage(const char* argv0) {
    std::ostringstream oss;
    oss
        << "usage: " << argv0 << " <model_path> [options]\n"
        << "\n"
        << "outputs:\n"
        << "  --emit-dot <path>     write DOT graph\n"
        << "  --emit-mlir <path>    write emitted MLIR\n"
        << "  --emit-llvm <path>    lower to LLVM IR\n"
        << "  --emit-asm <path>     lower to assembly\n"
        << "  --run                 execute graph with the built-in CPU runtime\n"
        << "  --input <name=path>   load whitespace/comma separated float32 input values\n"
        << "  --output-dir <dir>    write runtime outputs as text files\n"
        << "\n"
        << "llvm tuning:\n"
        << "  --target-triple <triple>\n"
        << "  --mcpu <cpu>\n"
        << "  --O0 | --O1 | --O2 | --O3\n";
    return oss.str();
}

DriverOptions ParseArgs(int argc, const char* argv[]) {
    DriverOptions opt;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--emit-dot") {
            opt.emit_dot_path = RequireValue(argc, argv, i, arg);
            continue;
        }
        if (arg == "--emit-mlir") {
            opt.emit_mlir_path = RequireValue(argc, argv, i, arg);
            continue;
        }
        if (arg == "--emit-llvm") {
            opt.emit_llvm_path = RequireValue(argc, argv, i, arg);
            continue;
        }
        if (arg == "--emit-asm") {
            opt.emit_asm_path = RequireValue(argc, argv, i, arg);
            continue;
        }
        if (arg == "--run") {
            opt.run = true;
            continue;
        }
        if (arg == "--input") {
            opt.input_paths.push_back(ParseInputSpec(RequireValue(argc, argv, i, arg)));
            continue;
        }
        if (arg == "--output-dir") {
            opt.output_dir = RequireValue(argc, argv, i, arg);
            continue;
        }
        if (arg == "--target-triple") {
            opt.target_triple = RequireValue(argc, argv, i, arg);
            continue;
        }
        if (arg == "--mcpu") {
            opt.mcpu = RequireValue(argc, argv, i, arg);
            continue;
        }
        if (arg == "--O0" || arg == "--O1" || arg == "--O2" || arg == "--O3") {
            opt.opt_level = std::string{"-O"} + arg.substr(3);
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error{"unknown flag: " + arg};
        }
        positional.push_back(arg);
    }

    if (positional.empty()) {
        throw std::runtime_error{"model path is required"};
    }
    if (positional.size() > 1) {
        throw std::runtime_error{"too many positional arguments"};
    }

    opt.model_path = positional[0];
    return opt;
}

} // namespace tc::driver
