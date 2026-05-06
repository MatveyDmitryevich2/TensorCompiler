#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "driver/driver_options.hpp"
#include "driver/tool_runner.hpp"
#include "graph/graph.hpp"
#include "mlir_backend/mlir_backend.hpp"
#include "onnx_loader/onnx_loader.hpp"
#include "runtime/interpreter.hpp"

namespace {

const tc::Value& RequireInputValue(const tc::Graph& graph, const std::string& name) {
    const tc::INode* node = graph.FindByName(name);
    const auto* value = dynamic_cast<const tc::Value*>(node);
    if (value == nullptr || !value->HasTensorType()) {
        throw std::runtime_error{"runtime input is not a typed graph value: " + name};
    }
    return *value;
}

tc::runtime::Tensor ReadInputTensor(const tc::Value& value, const std::string& path) {
    tc::runtime::Tensor tensor{value.MaybeTensorType()->Shape(), tc::runtime::ReadFloatTextFile(path)};
    if (tensor.data.size() != tensor.NumElements()) {
        throw std::runtime_error{"runtime input '" + value.Name() + "' element count mismatch"};
    }
    return tensor;
}

tc::runtime::TensorMap ReadRuntimeInputs(const tc::Graph& graph,
                                         const tc::driver::DriverOptions& opt) {
    tc::runtime::TensorMap inputs;
    for (const auto& [name, path] : opt.input_paths) {
        inputs.emplace(name, ReadInputTensor(RequireInputValue(graph, name), path));
    }
    return inputs;
}

void WriteRuntimeOutputs(const tc::runtime::TensorMap& outputs,
                         const std::string& output_dir) {
    for (const auto& [name, tensor] : outputs) {
        std::cout << name << " " << tc::runtime::ShapeToStr(tensor.shape)
                  << " elements=" << tensor.data.size() << '\n';
        if (!output_dir.empty()) {
            const std::filesystem::path out_path = std::filesystem::path{output_dir} / (name + ".txt");
            tc::runtime::WriteFloatTextFile(out_path.string(), tensor);
        }
    }
}

void RunGraph(const tc::Graph& graph, const tc::driver::DriverOptions& opt) {
    tc::runtime::Interpreter interpreter;
    WriteRuntimeOutputs(interpreter.Run(graph, ReadRuntimeInputs(graph, opt)), opt.output_dir);
}

} // namespace

int main(int argc, const char* argv[]) {
    tc::driver::SetupLogging(argc, argv);

    try {
        const tc::driver::DriverOptions opt = tc::driver::ParseArgs(argc, argv);

        tc::OnnxLoader loader;
        const tc::Graph graph = loader.Load(opt.model_path);

        if (!opt.emit_dot_path.empty()) {
            tc::driver::WriteTextFile(opt.emit_dot_path, graph.ToDot(tc::DotOptions{}));
        }

        if (opt.run) {
            RunGraph(graph, opt);
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
