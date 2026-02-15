#include "onnx_loader/onnx_loader.hpp"

#include <stdexcept>
#include <string>
#include <iostream>

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
        std::cout << "Input: " << i.name() << "\n";
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
    }

    for (const auto& o: g.output()) {
        std::cout << "Output: " << o.name() << "\n";
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

    return Graph{};
}

} // namespace tc
