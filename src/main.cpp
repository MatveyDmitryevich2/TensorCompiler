#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "driver/driver_options.hpp"
#include "driver/tool_runner.hpp"
#include "graph/graph.hpp"
#include "mlir_backend/mlir_backend.hpp"
#include "onnx_loader/onnx_loader.hpp"

int main(int argc, const char* argv[]) {
    tc::driver::SetupLogging(argc, argv);

    try {
        const tc::driver::DriverOptions opt = tc::driver::ParseArgs(argc, argv);

        tc::OnnxLoader loader;
        const tc::Graph graph = loader.Load(opt.model_path);

        if (!opt.emit_dot_path.empty()) {
            tc::driver::WriteTextFile(opt.emit_dot_path, graph.ToDot(tc::DotOptions{}));
        }

        std::string mlir_text;
        if (opt.NeedsMlir()) {
            tc::MlirBackend backend;
            mlir_text = backend.EmitModule(graph, tc::MlirEmitterOptions{});
        }

        if (!opt.emit_mlir_path.empty()) {
            tc::driver::WriteTextFile(opt.emit_mlir_path, mlir_text);
        }

        tc::driver::LowerToLlvmAndAsm(opt, mlir_text);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        std::cerr << tc::driver::Usage(argv[0]);
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}