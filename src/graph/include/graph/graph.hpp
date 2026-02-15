#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <vector>
#include <cstdint>

#include "graph/node.hpp"

namespace tc {

// class that owns nodes memory managies it
class NodeContainer {
  private:
    std::vector<INode*> nodes_;
  public:
    using const_iterator = std::vector<INode*>::const_iterator;

    NodeContainer() {}
 
// FIXME: implement AddNode method to add nodes to container
// FIXME: implement destructor that frees nodes

    INode* operator[](size_t idx) {
        return nodes_[idx];
    }

    const INode* operator[](size_t idx) const {
        return nodes_[idx];
    }

    const_iterator begin() const { return nodes_.begin(); }
    const_iterator end() const { return nodes_.end(); }
};

class Graph {
  private:
    NodeContainer nodes_;
  public:
    Graph() {}

    using const_iterator = NodeContainer::const_iterator;
    const_iterator begin() { return nodes_.begin(); }
    const_iterator end() { return nodes_.end(); }
};

} // namespace tc

#endif // GRAPH_HPP_
