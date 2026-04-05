#include "driver/tool_runner.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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

void RunCommand(const std::vector<std::string>& argv) {
    const std::string cmd = JoinCommand(argv);
    spdlog::info("run: {}", cmd);
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error{"command failed: " + cmd};
    }
}

} // namespace

void SetupLogging(int argc, const char* argv[]) {
    auto logger = spdlog::basic_logger_mt("tc", "tc.log", true);
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

    fs::path temp_dir = fs::temp_directory_path() / "tc_mlir_pipeline";
    fs::create_directories(temp_dir);

    const fs::path input_mlir = temp_dir / "input.mlir";
    const fs::path lowered_mlir = temp_dir / "lowered.mlir";
    const fs::path llvm_ir = temp_dir / "module.ll";
    const fs::path asm_file = temp_dir / "module.s";

    WriteTextFile(input_mlir.string(), mlir_text);

    std::vector<std::string> mlir_opt_cmd{kMlirOpt, input_mlir.string()};
    AppendLlvmLoweringPipeline(&mlir_opt_cmd);
    mlir_opt_cmd.push_back("-o");
    mlir_opt_cmd.push_back(lowered_mlir.string());
    RunCommand(mlir_opt_cmd);

    std::vector<std::string> mlir_translate_cmd{kMlirTranslate, lowered_mlir.string(), "--mlir-to-llvmir", "-o", llvm_ir.string()};
    RunCommand(mlir_translate_cmd);

    if (need_llvm) {
        CopyFileToTarget(llvm_ir, opt.emit_llvm_path);
    }

    if (need_asm) {
        std::vector<std::string> llc_cmd{kLlc, llvm_ir.string(), opt.opt_level};
        if (!opt.target_triple.empty()) {
            llc_cmd.push_back("-mtriple=" + opt.target_triple);
        }
        if (!opt.mcpu.empty()) {
            llc_cmd.push_back("-mcpu=" + opt.mcpu);
        }
        llc_cmd.push_back("-filetype=asm");
        llc_cmd.push_back("-o");
        llc_cmd.push_back(asm_file.string());
        RunCommand(llc_cmd);
        CopyFileToTarget(asm_file, opt.emit_asm_path);
    }

    std::error_code ec;
    fs::remove(input_mlir, ec);
    fs::remove(lowered_mlir, ec);
    fs::remove(llvm_ir, ec);
    fs::remove(asm_file, ec);
    fs::remove(temp_dir, ec);
}

} // namespace tc::driver