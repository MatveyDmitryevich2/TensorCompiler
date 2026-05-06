#include "onnx_loader/onnx_loader.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <string_view>
#include <vector>
#include <optional>
#include <cstring>

#include "onnx/onnx-ml.pb.h"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"

#include "helpers/trace_calls.hpp"
#include "graph/attribute.hpp"
#include "graph/node.hpp"
#include "graph/graph.hpp"
#include "graph/loader.hpp"

namespace tc {

namespace {

TensorElemType ParseElemType(int onnx_elem_type) {
    switch (onnx_elem_type) {
        case onnx::TensorProto_DataType_FLOAT:
            return TensorElemType::kFloat32;
        case onnx::TensorProto_DataType_DOUBLE:
            return TensorElemType::kFloat64;
        case onnx::TensorProto_DataType_INT32:
            return TensorElemType::kInt32;
        case onnx::TensorProto_DataType_INT64:
            return TensorElemType::kInt64;
        case onnx::TensorProto_DataType_BOOL:
            return TensorElemType::kBool;
        default:
            return TensorElemType::kUnknown;
    }
}

std::vector<int64_t> ParseShape(const onnx::TensorShapeProto& shape_proto) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(shape_proto.dim_size()));

    for (const auto& dim : shape_proto.dim()) {
        if (dim.has_dim_value()) {
            shape.push_back(static_cast<int64_t>(dim.dim_value()));
        } else {
            shape.push_back(-1);
        }
    }

    return shape;
}

std::optional<TensorType> ParseValueType(const onnx::ValueInfoProto& value_info) {
    if (!value_info.has_type()) return std::nullopt;
    if (!value_info.type().has_tensor_type()) return std::nullopt;

    const auto& tensor_type = value_info.type().tensor_type();
    TensorElemType elem_type = TensorElemType::kUnknown;
    if (tensor_type.has_elem_type()) {
        elem_type = ParseElemType(tensor_type.elem_type());
    }

    std::vector<int64_t> shape;
    if (tensor_type.has_shape()) {
        shape = ParseShape(tensor_type.shape());
    }

    return TensorType{elem_type, std::move(shape)};
}

template <typename T, typename RepeatedField>
std::string PackRepeatedData(const RepeatedField& values) {
    std::string raw(static_cast<size_t>(values.size()) * sizeof(T), '\0');
    if (!raw.empty()) {
        std::memcpy(raw.data(), values.data(), raw.size());
    }
    return raw;
}

TensorData ParseTensorData(const onnx::TensorProto& tensor) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(tensor.dims_size()));
    for (int i = 0; i < tensor.dims_size(); ++i) {
        shape.push_back(static_cast<int64_t>(tensor.dims(i)));
    }

    std::string raw;
    if (tensor.has_raw_data()) {
        raw = tensor.raw_data();
    } else if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT && tensor.float_data_size() != 0) {
        raw = PackRepeatedData<float>(tensor.float_data());
    } else if (tensor.data_type() == onnx::TensorProto_DataType_DOUBLE && tensor.double_data_size() != 0) {
        raw = PackRepeatedData<double>(tensor.double_data());
    } else if (tensor.data_type() == onnx::TensorProto_DataType_INT32 && tensor.int32_data_size() != 0) {
        raw = PackRepeatedData<int32_t>(tensor.int32_data());
    } else if (tensor.data_type() == onnx::TensorProto_DataType_INT64 && tensor.int64_data_size() != 0) {
        raw = PackRepeatedData<int64_t>(tensor.int64_data());
    }

    return TensorData{TensorType{ParseElemType(tensor.data_type()), std::move(shape)}, std::move(raw)};
}

Value* EnsureValue(Graph* graph, const std::string& name, Value::BelongTo belong) {
    if (name.empty()) return nullptr;

    Value* value = graph->FindValueByName(name);
    if (value == nullptr) {
        if (graph->Contains(name)) {
            throw std::runtime_error{"Expected Value node: " + name};
        }
        return graph->AddNode<Value>(name, belong);
    }

    value->UpgradeBelongsTo(belong);
    return value;
}

void MergeValueInfo(Graph* graph,
                    const onnx::ValueInfoProto& value_info,
                    Value::BelongTo belong) {
    if (value_info.name().empty()) {
        throw std::runtime_error{"ONNX ValueInfo has empty name"};
    }

    Value* value = EnsureValue(graph, value_info.name(), belong);
    std::optional<TensorType> type = ParseValueType(value_info);
    if (type.has_value()) {
        value->MergeTensorType(*type);
    }
}

AttributeMap ParseAttributes(const onnx::NodeProto& g_node) {
    AttributeMap out;

    for (const auto& a : g_node.attribute()) {
        const std::string& name = a.name();

        switch (a.type()) {
            case onnx::AttributeProto::INT:
                out.emplace(name, Attribute{name, static_cast<int64_t>(a.i())});
                break;

            case onnx::AttributeProto::FLOAT:
                out.emplace(name, Attribute{name, a.f()});
                break;

            case onnx::AttributeProto::STRING:
                out.emplace(name, Attribute{name, a.s()});
                break;

            case onnx::AttributeProto::INTS: {
                std::vector<int64_t> vec;
                vec.reserve(static_cast<size_t>(a.ints_size()));
                for (int i = 0; i < a.ints_size(); ++i) {
                    vec.push_back(static_cast<int64_t>(a.ints(i)));
                }
                out.emplace(name, Attribute{name, std::move(vec)});
                break;
            }

            case onnx::AttributeProto::FLOATS: {
                std::vector<float> vec;
                vec.reserve(static_cast<size_t>(a.floats_size()));
                for (int i = 0; i < a.floats_size(); ++i) {
                    vec.push_back(a.floats(i));
                }
                out.emplace(name, Attribute{name, std::move(vec)});
                break;
            }

            case onnx::AttributeProto::STRINGS: {
                std::vector<std::string> vec;
                vec.reserve(static_cast<size_t>(a.strings_size()));
                for (int i = 0; i < a.strings_size(); ++i) {
                    vec.push_back(a.strings(i));
                }
                out.emplace(name, Attribute{name, std::move(vec)});
                break;
            }

            default:
                throw std::runtime_error{"Unsupported ONNX attribute type: '" + name + "'"};
        }
    }

    return out;
}

Operation::OpType StrToOp(const std::string& op_type) {
    static const std::unordered_map<std::string_view, Operation::OpType> str_to_op = {
        {"Add",       Operation::OpType::kAdd      },
        {"MatMul",    Operation::OpType::kMatMul   },
        {"Transpose", Operation::OpType::kTranspose},
        {"Mul",       Operation::OpType::kMul      },
        {"Conv",      Operation::OpType::kConv     },
        {"Relu",      Operation::OpType::kRelu     },
        {"Gemm",      Operation::OpType::kGemm     },
    };

    auto it = str_to_op.find(op_type);
    if (it == str_to_op.end()) {
        throw std::runtime_error{"Unsupported op_type: " + op_type};
    }
    return it->second;
}

std::string MakeOperationName(const Graph& graph, const onnx::NodeProto& g_node, size_t index) {
    if (!g_node.name().empty()) {
        return g_node.name();
    }

    const std::string prefix = g_node.op_type().empty() ? "op" : g_node.op_type();
    std::string candidate = prefix + "_" + std::to_string(index);
    size_t suffix = 0;
    while (graph.FindByName(candidate) != nullptr) {
        candidate = prefix + "_" + std::to_string(index) + "_" + std::to_string(++suffix);
    }
    return candidate;
}

void AddOpNode(Graph* graph, const onnx::NodeProto& g_node, size_t index) {
    const std::string name = MakeOperationName(*graph, g_node, index);
    if (graph->FindByName(name) != nullptr) {
        throw std::runtime_error{"Duplicate ONNX node name: " + name};
    }

    Operation::OpType op = StrToOp(g_node.op_type());

    std::vector<Value*> inputs;
    std::vector<Value*> outputs;

    for (const auto& input_name : g_node.input()) {
        if (!input_name.empty()) {
            inputs.push_back(EnsureValue(graph, input_name, Value::BelongTo::kInternal));
        }
    }
    for (const auto& output_name : g_node.output()) {
        if (!output_name.empty()) {
            outputs.push_back(EnsureValue(graph, output_name, Value::BelongTo::kInternal));
        }
    }

    AttributeMap attrs = ParseAttributes(g_node);
    graph->AddNode<Operation>(name, op, inputs, outputs, attrs);
}

} // namespace

Graph OnnxLoader::ParseRaw(const std::string& model_raw) {
    hlp::trace_call();

    onnx::ModelProto model;
    bool success = model.ParseFromString(model_raw);
    if (!success) {
        throw std::runtime_error{"Unable to parse onnx model"};
    }

    Graph graph;
    const onnx::GraphProto& onnx_graph = model.graph();

    for (const onnx::ValueInfoProto& input : onnx_graph.input()) {
        MergeValueInfo(&graph, input, Value::BelongTo::kInput);
    }

    for (const onnx::ValueInfoProto& output : onnx_graph.output()) {
        MergeValueInfo(&graph, output, Value::BelongTo::kOutput);
    }

    for (const onnx::ValueInfoProto& value_info : onnx_graph.value_info()) {
        MergeValueInfo(&graph, value_info, Value::BelongTo::kInternal);
    }

    for (const onnx::TensorProto& init_tensor : onnx_graph.initializer()) {
        TensorData tensor_data = ParseTensorData(init_tensor);
        Value* value = EnsureValue(&graph, init_tensor.name(), Value::BelongTo::kInitializer);
        value->UpgradeBelongsTo(Value::BelongTo::kInitializer);
        value->MergeInitializerData(std::move(tensor_data));
    }

    for (const onnx::NodeProto& g_node : onnx_graph.node()) {
        for (const std::string& input_name : g_node.input()) {
            if (!input_name.empty()) {
                EnsureValue(&graph, input_name, Value::BelongTo::kInternal);
            }
        }
        for (const std::string& output_name : g_node.output()) {
            if (!output_name.empty()) {
                EnsureValue(&graph, output_name, Value::BelongTo::kInternal);
            }
        }
    }

    for (int i = 0; i < onnx_graph.node_size(); ++i) {
        AddOpNode(&graph, onnx_graph.node(i), static_cast<size_t>(i));
    }

    return graph;
}

} // namespace tc
