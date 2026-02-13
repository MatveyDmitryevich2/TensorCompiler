#include "onnx_loader/onnx_loader.hpp"

#include <string>
#include <iostream>

#include "graph/graph.hpp"
#include "graph/loader.hpp"

namespace tc {

Graph OnnxLoader::ParseRaw(const std::string& model_raw) {
    std::cout << model_raw; // FIXME

    return Graph{};
}

} // namespace tc
