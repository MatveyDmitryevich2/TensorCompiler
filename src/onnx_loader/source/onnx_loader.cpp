#include "onnx_loader/onnx_loader.hpp"

#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>
#include <string_view>

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
                std::vector<int64_t> v;
                v.reserve(static_cast<size_t>(a.ints_size()));
                for (int i = 0; i < a.ints_size(); ++i) v.push_back(static_cast<int64_t>(a.ints(i)));
                out.emplace(name, Attribute{name, std::move(v)});
                break;
            }
            case onnx::AttributeProto::FLOATS: {
                std::vector<float> v;
                v.reserve(static_cast<size_t>(a.floats_size()));
                for (int i = 0; i < a.floats_size(); ++i) v.push_back(a.floats(i));
                out.emplace(name, Attribute{name, std::move(v)});
                break;
            }
            case onnx::AttributeProto::STRINGS: {
                std::vector<std::string> v;
                v.reserve(static_cast<size_t>(a.strings_size()));
                for (int i = 0; i < a.strings_size(); ++i) v.push_back(a.strings(i));
                out.emplace(name, Attribute{name, std::move(v)});
                break;
            }
            default:
                throw std::runtime_error{"Unsupported ONNX attribute type: '" + name + "'"};
        }
    }

    return out;
}


// FIXME: implement dot dump
void Dump(const onnx::ModelProto& model) {
    const auto& g = model.graph();
    for (const auto& i: g.input()) {
        std::cout << "Value: " << i.name() << "\n";
        if (i.has_type() && i.type().has_tensor_type()) {
            const auto& shape = i.type().tensor_type().shape();
            for (const auto& d: shape.dim()) {
                std::cout << "    Shape: " 
                    << (d.has_dim_value() 
                        ? std::to_string(d.dim_value()) 
                        : "?") 
                    << "\n";
            }
        }
    }

    std::cout << "Graph:" << "\n";

    for (const auto& n: g.node()) {
        std::cout << "    Name: " << n.name() << "\n";
        std::cout << "    Type: " << n.op_type() << "\n";

        for (const auto& i: n.input()) {
            std::cout << "        Input: " << i << "\n";
        }

        for (const auto& o: n.output()) {
            std::cout << "        Output: " << o << "\n";
        }

        for (const auto& a: n.attribute()) {
            std::cout << "        Attribute: " << a.name() << "\n";
        }
    }

    for (const auto& o: g.output()) {
        std::cout << "Value: " << o.name() << "\n";
        if (o.has_type() && o.type().has_tensor_type()) {
            const auto& shape = o.type().tensor_type().shape();
            for (const auto& d: shape.dim()) {
                std::cout << "    Shape: " 
                    << (d.has_dim_value() 
                        ? std::to_string(d.dim_value()) 
                        : "?") 
                    << "\n";
            }
        }
    }
}

OpType StrToOp(const std::string& op_type) {
    std::unordered_map<std::string_view, OpType> str_to_op = {
        {"Add",       OpType::kAdd      },
        {"MatMul",    OpType::kMatMul   },
        {"Transpose", OpType::kTranspose},
        {"Mul",       OpType::kMul      },
        {"Conv",      OpType::kConv     },
        {"Relu",      OpType::kRelu     },
        {"Gemm",      OpType::kGemm     },
    };

    auto it = str_to_op.find(op_type);
    if (it == str_to_op.end()) {
        throw std::runtime_error{"Unsupported op_type: " + op_type};
    }
    return it->second;
}

void AddOpNode(Graph* graph, const onnx::NodeProto& g_node) {
    OpType op = StrToOp(g_node.op_type());
    std::string name = g_node.name();

    auto find = [graph](const std::string& name) {
        return static_cast<Value*>(graph->FindByName(name));
    };

    std::vector<Value*> inputs;
    std::vector<Value*> outputs;

    for (const auto& i : g_node.input()) {
        inputs.push_back(find(i));
    }
    for (const auto& o : g_node.output()) {
        outputs.push_back(find(o));
    }
    
    AttributeMap attrs = ParseAttributes(g_node);
    graph->AddNode<Operation>(name, op, inputs, outputs, attrs);

}

} // namaspace

Graph OnnxLoader::ParseRaw(const std::string& model_raw) {
    hlp::trace_call();

    onnx::ModelProto model;
    bool success = model.ParseFromString(model_raw);
    if (!success) {
        throw std::runtime_error{"Unable to parse onnx model"};
    }

    Dump(model);

    // FIXME: convert onnx model to graph

    Graph graph;

    const onnx::GraphProto& onnx_graph = model.graph();
    for (const onnx::ValueInfoProto& g_input: onnx_graph.input()) {
        graph.AddNode<Value>(g_input.name(), Value::BelongTo::kInput);
    }

    for (const onnx::ValueInfoProto& g_output: onnx_graph.output()) {
        graph.AddNode<Value>(g_output.name(), Value::BelongTo::kOutput);
    }

    for (const onnx::TensorProto& t : onnx_graph.initializer()) {
        Value* v = graph.AddNode<Value>(t.name(), Value::BelongTo::kInitializer);

        v->SetBelongsTo(Value::BelongTo::kInitializer);

        if (t.has_raw_data()) {
            TensorData data;
            const std::string& raw = t.raw_data();
            data.raw.assign(raw.begin(), raw.end());
            v->SetData(std::move(data));
        }
    }

    for (const onnx::NodeProto& g_node: onnx_graph.node()) {
        for (const std::string& n_input: g_node.input()) {
            graph.AddNode<Value>(n_input, Value::BelongTo::kInternal);
        }
        for (const std::string& n_output: g_node.output()) {
            graph.AddNode<Value>(n_output, Value::BelongTo::kInternal);
        }
    }

    for (const onnx::NodeProto& g_node: onnx_graph.node()) {
        AddOpNode(&graph, g_node);
    }

    return graph;
}

} // namespace tc