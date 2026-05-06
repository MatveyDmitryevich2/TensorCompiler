#include "driver/tool_runner.hpp"

#include "tool_runner_internal.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace tc::driver {

namespace {

constexpr const char* kMlirOpt = "mlir-opt";
constexpr const char* kMlirTranslate = "mlir-translate";
constexpr const char* kLlc = "llc";
constexpr const char* kLogPath = "run_data/logs/tc.log";

void AppendLlvmLoweringPipeline(std::vector<std::string>* cmd) {
    cmd->push_back("--canonicalize");
    cmd->push_back("--cse");
    cmd->push_back("--convert-scf-to-cf");
    cmd->push_back("--expand-strided-metadata");
    cmd->push_back("--convert-index-to-llvm");
    cmd->push_back("--convert-arith-to-llvm");
    cmd->push_back("--convert-func-to-llvm");
    cmd->push_back("--finalize-memref-to-llvm");
    cmd->push_back("--convert-cf-to-llvm");
    cmd->push_back("--reconcile-unrealized-casts");
}

std::string ShellQuote(const std::string& value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

std::string JoinCommand(const std::vector<std::string>& argv) {
    std::string out;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) {
            out += ' ';
        }
        out += ShellQuote(argv[i]);
    }
    return out;
}

void CopyFileToTarget(const fs::path& src, const std::string& dst) {
    WriteTextFile(dst, ReadTextFile(src));
}

fs::path MakeTempDir(std::string_view prefix) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path dir = fs::temp_directory_path() / (std::string(prefix) + "_" + std::to_string(ticks));
    fs::create_directories(dir);
    return dir;
}

} // namespace

void RunCommand(const std::vector<std::string>& argv) {
    const std::string cmd = JoinCommand(argv);
    spdlog::info("run: {}", cmd);
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error{"command failed: " + cmd};
    }
}

void RemoveTree(const fs::path& path) {
    std::error_code ec;
    fs::remove_all(path, ec);
}

std::vector<std::string> BuildLlcCommand(const DriverOptions& opt,
                                         const fs::path& input,
                                         std::string_view filetype,
                                         const fs::path& output) {
    std::vector<std::string> cmd{kLlc, input.string(), opt.opt_level};
    if (!opt.target_triple.empty()) {
        cmd.push_back("-mtriple=" + opt.target_triple);
    }
    if (!opt.mcpu.empty()) {
        cmd.push_back("-mcpu=" + opt.mcpu);
    }
    cmd.push_back("-filetype=" + std::string(filetype));
    cmd.push_back("-o");
    cmd.push_back(output.string());
    return cmd;
}

LoweredPaths LowerMlirToLlvm(const std::string& mlir_text) {
    LoweredPaths paths;
    paths.temp_dir = MakeTempDir("tc_mlir_pipeline");
    paths.input_mlir = paths.temp_dir / "input.mlir";
    paths.lowered_mlir = paths.temp_dir / "lowered.mlir";
    paths.llvm_ir = paths.temp_dir / "module.ll";

    WriteTextFile(paths.input_mlir.string(), mlir_text);

    std::vector<std::string> mlir_opt_cmd{kMlirOpt, paths.input_mlir.string()};
    AppendLlvmLoweringPipeline(&mlir_opt_cmd);
    mlir_opt_cmd.push_back("-o");
    mlir_opt_cmd.push_back(paths.lowered_mlir.string());
    RunCommand(mlir_opt_cmd);

    std::vector<std::string> mlir_translate_cmd{
        kMlirTranslate,
        paths.lowered_mlir.string(),
        "--mlir-to-llvmir",
        "-o",
        paths.llvm_ir.string()
    };
    RunCommand(mlir_translate_cmd);
    return paths;
}

void SetupLogging(int argc, const char* argv[]) {
    fs::create_directories(fs::path{kLogPath}.parent_path());
    auto logger = spdlog::basic_logger_mt("tc", kLogPath, true);
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%l] %v");

#if defined(NDEBUG)
    spdlog::set_level(spdlog::level::info);
#else
    spdlog::flush_on(spdlog::level::trace);
    spdlog::set_level(spdlog::level::trace);
#endif

    for (int i = 0; i < argc; ++i) {
        spdlog::info("argv[{}]: {}", i, argv[i]);
    }
}

std::string ReadTextFile(const fs::path& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in.is_open()) {
        throw std::runtime_error{"unable to open file for reading: " + path.string()};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void WriteTextFile(const std::string& path, const std::string& text) {
    if (path == "-") {
        std::cout << text;
        return;
    }

    const fs::path parent = fs::path{path}.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }

    std::ofstream out{path, std::ios::binary};
    if (!out.is_open()) {
        throw std::runtime_error{"unable to open file for writing: " + path};
    }
    out << text;
    spdlog::info("wrote: {}", path);
}

void LowerToLlvmAndAsm(const DriverOptions& opt, const std::string& mlir_text) {
    const bool need_llvm = !opt.emit_llvm_path.empty();
    const bool need_asm = !opt.emit_asm_path.empty();
    if (!need_llvm && !need_asm) {
        return;
    }

    LoweredPaths paths = LowerMlirToLlvm(mlir_text);

    if (need_llvm) {
        CopyFileToTarget(paths.llvm_ir, opt.emit_llvm_path);
    }

    if (need_asm) {
        const fs::path asm_file = paths.temp_dir / "module.s";
        RunCommand(BuildLlcCommand(opt, paths.llvm_ir, "asm", asm_file));
        CopyFileToTarget(asm_file, opt.emit_asm_path);
    }

    RemoveTree(paths.temp_dir);
}

} // namespace tc::driver
