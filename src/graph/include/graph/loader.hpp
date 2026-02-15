#ifndef LOADER_HPP_
#define LOADER_HPP_

#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>

#include "graph/graph.hpp"

namespace tc {

class ILoader {
  public:
    virtual ~ILoader() = default;
  private:
    virtual Graph ParseRaw(const std::string& model_raw) = 0;
  public:
    Graph Load(const std::string& model_path) {
        std::stringstream buffer;
        {
            std::ifstream model_file{model_path, std::ios::binary};
            if (!model_file.good()) {
                throw std::runtime_error{"Unable to open model file"};
            }
            buffer << model_file.rdbuf();
        }
        std::string model_raw = buffer.str();

        return ParseRaw(model_raw);
    }
};

} // namespace tc

#endif // LOADER_HPP_
