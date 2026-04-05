#ifndef MLIR_BACKEND_HPP_
#define MLIR_BACKEND_HPP_

#include <string>

#include "graph/graph.hpp"

namespace tc {

struct MlirEmitterOptions {
    std::string entry_name = "main";
};

class MlirBackend {
  public:
    std::string EmitModule(const Graph& graph, const MlirEmitterOptions& options = {}) const;
};

} // namespace tc

#endif // MLIR_BACKEND_HPP_
