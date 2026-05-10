#include "driver/aot_runner.hpp"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "driver/tool_runner.hpp"
#include "graph/node.hpp"
#include "tool_runner_internal.hpp"

namespace fs = std::filesystem;

namespace tc::driver {

namespace {

std::string CxxStringLiteral(std::string_view value) {
    std::string out = "\"";
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    out += "\"";
    return out;
}

std::string CxxIdentifier(std::string_view name) {
    std::string out;
    out.reserve(name.size() + 2);
    for (char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_';
        out += ok ? c : '_';
    }
    if (out.empty() || (out[0] >= '0' && out[0] <= '9')) {
        out = "v_" + out;
    }
    return out;
}

size_t NumElements(const std::vector<int64_t>& shape) {
    size_t total = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error{"AOT runner does not support dynamic shapes"};
        }
        total *= static_cast<size_t>(dim);
    }
    return total;
}

std::vector<int64_t> RowMajorStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    return strides;
}

const TensorType& RequireFloat32Tensor(const Value& value) {
    if (!value.HasTensorType()) {
        throw std::runtime_error{"AOT value has no tensor type: " + value.Name()};
    }
    const TensorType& type = *value.MaybeTensorType();
    if (type.ElemType() != TensorElemType::kFloat32) {
        throw std::runtime_error{"AOT runner supports float32 graph inputs/outputs only: " + value.Name()};
    }
    (void)NumElements(type.Shape());
    return type;
}

const std::string& InputPathFor(const DriverOptions& opt, std::string_view name) {
    for (const auto& [input_name, path] : opt.input_paths) {
        if (input_name == name) {
            return path;
        }
    }
    throw std::runtime_error{"missing input path for compiled run: " + std::string{name}};
}

void AppendMemRefShapeSuffix(std::vector<std::string>* values,
                             const std::string& base,
                             const std::string& suffix,
                             const std::vector<int64_t>& shape) {
    for (size_t i = 0; i < shape.size(); ++i) {
        values->push_back("int64_t " + base + suffix + std::to_string(i));
    }
}

void AppendMemRefSignature(std::vector<std::string>* params, const Value& value) {
    const TensorType& type = RequireFloat32Tensor(value);
    const std::string base = CxxIdentifier(value.Name());
    params->push_back("float* " + base + "_allocated");
    params->push_back("float* " + base + "_aligned");
    params->push_back("int64_t " + base + "_offset");
    AppendMemRefShapeSuffix(params, base, "_size", type.Shape());
    AppendMemRefShapeSuffix(params, base, "_stride", type.Shape());
}

void AppendInts(std::vector<std::string>* values, const std::vector<int64_t>& ints) {
    for (int64_t value : ints) {
        values->push_back(std::to_string(value));
    }
}

std::string JoinComma(const std::vector<std::string>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ", ";
        out += values[i];
    }
    return out;
}

void AppendMemRefCallArgs(std::vector<std::string>* args, const Value& value) {
    const TensorType& type = RequireFloat32Tensor(value);
    const std::string var = CxxIdentifier(value.Name());

    args->push_back(var + ".data()");
    args->push_back(var + ".data()");
    args->push_back("0");
    AppendInts(args, type.Shape());
    AppendInts(args, RowMajorStrides(type.Shape()));
}

std::string BuildAotRunnerSource(const Graph& graph, const DriverOptions& opt) {
    const std::vector<const Value*> inputs = graph.ValuesByBelong(Value::BelongTo::kInput);
    const std::vector<const Value*> outputs = graph.ValuesByBelong(Value::BelongTo::kOutput);
    if (outputs.empty()) {
        throw std::runtime_error{"compiled run requires graph outputs"};
    }

    std::vector<std::string> signature;
    for (const Value* value : inputs) {
        AppendMemRefSignature(&signature, *value);
    }
    for (const Value* value : outputs) {
        AppendMemRefSignature(&signature, *value);
    }

    std::vector<std::string> call_args;
    for (const Value* value : inputs) {
        AppendMemRefCallArgs(&call_args, *value);
    }
    for (const Value* value : outputs) {
        AppendMemRefCallArgs(&call_args, *value);
    }

    std::ostringstream src;
    src
        << "#include <cstdint>\n"
        << "#include <filesystem>\n"
        << "#include <fstream>\n"
        << "#include <iomanip>\n"
        << "#include <iostream>\n"
        << "#include <limits>\n"
        << "#include <sstream>\n"
        << "#include <stdexcept>\n"
        << "#include <string>\n"
        << "#include <vector>\n\n"
        << "extern \"C\" void entry_main(" << JoinComma(signature) << ");\n\n"
        << "std::vector<float> ReadTensor(const std::string& path, size_t count) {\n"
        << "    std::ifstream in{path};\n"
        << "    if (!in.is_open()) throw std::runtime_error{\"unable to open input: \" + path};\n"
        << "    std::vector<float> values;\n"
        << "    std::string token;\n"
        << "    while (in >> token) {\n"
        << "        for (char& c : token) if (c == ',') c = ' ';\n"
        << "        std::istringstream one{token};\n"
        << "        float value = 0.0f;\n"
        << "        while (one >> value) values.push_back(value);\n"
        << "    }\n"
        << "    if (values.size() != count) throw std::runtime_error{\"input element count mismatch: \" + path};\n"
        << "    return values;\n"
        << "}\n\n"
        << "void WriteTensor(const std::string& path, const std::vector<float>& values) {\n"
        << "    const std::filesystem::path parent = std::filesystem::path{path}.parent_path();\n"
        << "    if (!parent.empty()) std::filesystem::create_directories(parent);\n"
        << "    std::ofstream out{path};\n"
        << "    if (!out.is_open()) throw std::runtime_error{\"unable to open output: \" + path};\n"
        << "    for (size_t i = 0; i < values.size(); ++i) {\n"
        << "        if (i != 0) out << ' ';\n"
        << "        out << std::setprecision(std::numeric_limits<float>::max_digits10) << values[i];\n"
        << "    }\n"
        << "    out << '\\n';\n"
        << "}\n\n"
        << "int main() {\n"
        << "  try {\n";

    for (const Value* value : inputs) {
        const TensorType& type = RequireFloat32Tensor(*value);
        src << "    std::vector<float> " << CxxIdentifier(value->Name())
            << " = ReadTensor(" << CxxStringLiteral(InputPathFor(opt, value->Name()))
            << ", " << NumElements(type.Shape()) << ");\n";
    }
    for (const Value* value : outputs) {
        const TensorType& type = RequireFloat32Tensor(*value);
        src << "    std::vector<float> " << CxxIdentifier(value->Name())
            << "(" << NumElements(type.Shape()) << ", 0.0f);\n";
    }

    src << "    entry_main(" << JoinComma(call_args) << ");\n";

    for (const Value* value : outputs) {
        const std::string output_path = opt.output_dir.empty()
            ? std::string{}
            : (fs::path{opt.output_dir} / (value->Name() + ".txt")).string();
        src << "    std::cout << " << CxxStringLiteral(value->Name()) << " << \" elements=\" << "
            << CxxIdentifier(value->Name()) << ".size() << '\\n';\n";
        if (!output_path.empty()) {
            src << "    WriteTensor(" << CxxStringLiteral(output_path) << ", "
                << CxxIdentifier(value->Name()) << ");\n";
        }
    }

    src
        << "  } catch (const std::exception& e) {\n"
        << "    std::cerr << e.what() << '\\n';\n"
        << "    return 1;\n"
        << "  }\n"
        << "  return 0;\n"
        << "}\n";
    return src.str();
}

std::string CxxCompiler() {
    const char* cxx = std::getenv("CXX");
    if (cxx != nullptr && cxx[0] != '\0') {
        return cxx;
    }
    return "c++";
}

} // namespace

void CompileAndRunAot(const Graph& graph, const DriverOptions& opt, const std::string& mlir_text) {
    LoweredPaths lowered = LowerMlirToLlvm(mlir_text);
    const fs::path runner_cpp = lowered.temp_dir / "runner.cpp";
    const fs::path module_obj = lowered.temp_dir / "module.o";
    const fs::path runner_exe = lowered.temp_dir / "runner.x";

    WriteTextFile(runner_cpp.string(), BuildAotRunnerSource(graph, opt));

    RunCommand(BuildLlcCommand(opt, lowered.llvm_ir, "obj", module_obj));

    RunCommand({CxxCompiler(), "-std=c++17", runner_cpp.string(), module_obj.string(), "-O2", "-o", runner_exe.string()});
    RunCommand({runner_exe.string()});
    RemoveTree(lowered.temp_dir);
}

} // namespace tc::driver
