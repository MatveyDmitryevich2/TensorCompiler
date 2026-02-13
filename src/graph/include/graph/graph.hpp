#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <vector>
#include <cstdint>

namespace tc {

class INode {
  public:
    using Id = int64_t;
    using AdjList = std::vector<Id>;
    AdjList adj_nodes_;

    virtual ~INode() = default;
};

class NodeContainer {
  private:
    std::vector<INode*> nodes_;
  public:
    NodeContainer() {}
    
    INode* operator[](size_t idx) {
        return nodes_[idx];
    }

    const INode* operator[](size_t idx) const {
        return nodes_[idx];
    }
};

class Graph {
  private:
    NodeContainer nodes_;
  public:
    Graph() {}
};

} // namespace tc

#endif // GRAPH_HPP_
