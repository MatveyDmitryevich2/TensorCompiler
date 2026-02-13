#ifndef ONNX_LOADER_HPP_
#define ONNX_LOADER_HPP_

#include <string>
#include <fstream>
#include <sstream>

#include "graph/loader.hpp"

namespace tc {

class OnnxLoader : public ILoader {
  public:
    ~OnnxLoader() override = default;
    Graph Load(const std::string& model_path) override {
        std::stringstream buffer;
        {
            std::ifstream model_file{model_path, std::ios::binary};
            buffer << buffer.rdbuf();
        }
        std::string model_raw = buffer.str();

        return ParseRaw(model_raw);
    }
  private:
    Graph ParseRaw(const std::string& model_raw);
};

} // namespace tc

#endif // ONNX_LOADER_HPP_
