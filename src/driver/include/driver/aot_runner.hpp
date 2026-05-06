#ifndef AOT_RUNNER_HPP_
#define AOT_RUNNER_HPP_

#include <string>

#include "driver/driver_options.hpp"
#include "graph/graph.hpp"

namespace tc::driver {

void CompileAndRunAot(const Graph& graph, const DriverOptions& opt, const std::string& mlir_text);

} // namespace tc::driver

#endif // AOT_RUNNER_HPP_
