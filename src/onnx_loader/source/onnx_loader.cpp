#include "onnx_loader/onnx_loader.hpp"

#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>
#include <string_view>

#include "graph/node.hpp"
#include "onnx/onnx-ml.pb.h"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"

#include "helpers/trace_calls.hpp"
#include "graph/graph.hpp"
#include "graph/loader.hpp"

namespace tc {

namespace {

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
    graph->AddNode<Node>(name, op, inputs, outputs, attrs);

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
