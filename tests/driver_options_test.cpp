#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "driver/driver_options.hpp"

namespace {

tc::driver::DriverOptions Parse(std::vector<const char*> argv) {
    return tc::driver::ParseArgs(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST(driver_options, ParsesRuntimeAndOutputOptions) {
    const tc::driver::DriverOptions opt = Parse({
        "tc.x",
        "model.onnx",
        "--run",
        "--input", "X=inputs/x.txt",
        "--input", "A=inputs/a.txt",
        "--output-dir", "outputs",
        "--emit-dot", "graph.dot",
        "--emit-mlir", "module.mlir"
    });

    EXPECT_EQ(opt.model_path, "model.onnx");
    EXPECT_TRUE(opt.run);
    EXPECT_FALSE(opt.run_compiled);
    ASSERT_EQ(opt.input_paths.size(), 2U);
    EXPECT_EQ(opt.input_paths[0], (std::pair<std::string, std::string>{"X", "inputs/x.txt"}));
    EXPECT_EQ(opt.input_paths[1], (std::pair<std::string, std::string>{"A", "inputs/a.txt"}));
    EXPECT_EQ(opt.output_dir, "outputs");
    EXPECT_EQ(opt.emit_dot_path, "graph.dot");
    EXPECT_EQ(opt.emit_mlir_path, "module.mlir");
    EXPECT_TRUE(opt.NeedsMlir());
}

TEST(driver_options, ParsesCompilationTuningOptions) {
    const tc::driver::DriverOptions opt = Parse({
        "tc.x",
        "model.onnx",
        "--run-compiled",
        "--emit-llvm", "module.ll",
        "--emit-asm", "module.s",
        "--target-triple", "x86_64-pc-linux-gnu",
        "--mcpu", "native",
        "--O3"
    });

    EXPECT_FALSE(opt.run);
    EXPECT_TRUE(opt.run_compiled);
    EXPECT_EQ(opt.emit_llvm_path, "module.ll");
    EXPECT_EQ(opt.emit_asm_path, "module.s");
    EXPECT_EQ(opt.target_triple, "x86_64-pc-linux-gnu");
    EXPECT_EQ(opt.mcpu, "native");
    EXPECT_EQ(opt.opt_level, "-O3");
    EXPECT_TRUE(opt.NeedsMlir());
}

TEST(driver_options, RejectsInvalidArguments) {
    EXPECT_THROW(Parse({"tc.x"}), std::runtime_error);
    EXPECT_THROW(Parse({"tc.x", "a.onnx", "b.onnx"}), std::runtime_error);
    EXPECT_THROW(Parse({"tc.x", "model.onnx", "--unknown"}), std::runtime_error);
    EXPECT_THROW(Parse({"tc.x", "model.onnx", "--input", "X"}), std::runtime_error);
    EXPECT_THROW(Parse({"tc.x", "model.onnx", "--emit-dot"}), std::runtime_error);
}
