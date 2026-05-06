#ifndef RUNTIME_INTERPRETER_HPP_
#define RUNTIME_INTERPRETER_HPP_

#include <string>
#include <unordered_map>

#include "graph/graph.hpp"
#include "runtime/tensor.hpp"

namespace tc::runtime {

using TensorMap = std::unordered_map<std::string, Tensor>;

class Interpreter {
  public:
    TensorMap Run(const Graph& graph, const TensorMap& inputs) const;
};

} // namespace tc::runtime

#endif // RUNTIME_INTERPRETER_HPP_
