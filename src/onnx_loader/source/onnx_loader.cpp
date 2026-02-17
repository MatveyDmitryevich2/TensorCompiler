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

enum Operations {
    kAdd,
    kMatMul,
    kTranspose,
    kMul,
    kConv,
    kRelu,
    kGemm,
};

Operations StrToOp(const std::string& op_type) {
    std::unordered_map<std::string_view, Operations> str_to_op = {
        {"Add", kAdd},
        {"MatMul", kMatMul},
        {"Transpose", kTranspose},
        {"Mul", kMul},
        {"Conv", kConv},
        {"Relu", kRelu},
        {"Gemm", kGemm},
    };

    return str_to_op[op_type];
}

void AddOpNode(Graph* graph, const onnx::NodeProto& g_node) {
    Operations op = StrToOp(g_node.op_type());
    std::string name = g_node.name();

    auto find = [graph](const std::string& name) {
        return static_cast<Value*>(graph->FindByName(name));
    };

    switch (op) {
        case kAdd:
            graph->AddNode<Add>(
                name,
                find(g_node.input()[0]),
                find(g_node.input()[1]),
                find(g_node.output()[0])
            );
            break;
        case kMatMul:
            graph->AddNode<MatMul>(
                name,
                find(g_node.input()[0]),
                find(g_node.input()[1]),
                find(g_node.output()[0])
            );
            break;
        case kTranspose:
            graph->AddNode<Transpose>(
                name,
                find(g_node.input()[0]),
                find(g_node.output()[0])
            );
            break;
    }
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
