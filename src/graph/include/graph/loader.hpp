#ifndef LOADER_HPP_
#define LOADER_HPP_

#include <string>

#include "graph/graph.hpp"

namespace tc {

class ILoader {
  public:
    virtual ~ILoader() = default;
    
    virtual Graph Load(const std::string& model_path) = 0;
};

} // namespace tc

#endif // LOADER_HPP_
