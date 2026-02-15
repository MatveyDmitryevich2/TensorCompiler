#ifndef ONNX_LOADER_HPP_
#define ONNX_LOADER_HPP_

#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>

#include "helpers/trace_calls.hpp"
#include "graph/loader.hpp"

namespace tc {

class OnnxLoader : public ILoader {
  public:
    ~OnnxLoader() override = default;
  private:
    Graph ParseRaw(const std::string& model_raw) override;
};

} // namespace tc

#endif // ONNX_LOADER_HPP_
