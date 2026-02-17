#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <memory>
#include <utility>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <unordered_map>

#include <spdlog/spdlog.h>
#include "graph/node.hpp"

namespace tc {

// class that owns nodes memory managies it
class NodeContainer {
  private:
    using NodesOwner = std::vector<INode*>;
    using NameTable = std::unordered_map<std::string, INode*>;

    NodesOwner nodes_;
    NameTable name_table_;

  public:
    using const_iterator = std::vector<INode*>::const_iterator;

    // FIXME: implement rule of 5
    NodeContainer() {}
    NodeContainer(const NodeContainer& other) = delete;
    NodeContainer& operator=(const NodeContainer& other) = delete;
    NodeContainer(NodeContainer&& other) = default;
    NodeContainer& operator=(NodeContainer&& other) = default;

    ~NodeContainer() {
        for (size_t i = 0; i < nodes_.size(); i++) {
            delete nodes_[i];
        }
    }

    template <typename NodeT, typename... Args> 
    NodeT* AddNode(const std::string& name, Args&&... args) {
        static_assert(std::is_base_of_v<INode, NodeT>, "NodeT should be derived from INode");
        
        if constexpr (std::is_same_v<NodeT, Value>) { // Value nodes can be tried to added multiple times
            auto&& node_it = name_table_.find(name);
            if (node_it != name_table_.end()) {
                return static_cast<NodeT*>(node_it->second);
            }
        }

        std::unique_ptr<NodeT> node = std::make_unique<NodeT>(name, std::forward<Args>(args)...);
        NodeT* raw_ptr = node.get();

        NodesOwner tmp_owner{nodes_};
        NameTable tmp_table{name_table_};

        tmp_owner.push_back(raw_ptr);
        tmp_table.insert({name, raw_ptr});

        // commit
        std::swap(nodes_, tmp_owner);
        std::swap(name_table_, tmp_table);

        node.release();

        return raw_ptr;
    }

    bool Contains(const std::string& name) {
        return name_table_.contains(name);
    }

    INode* FindByName(const std::string& name) {
        auto&& node = name_table_.find(name);
        if (node == name_table_.end()) { 
            SPDLOG_TRACE("Not found {}", name);
            return nullptr; 
        }

        return node->second;
    }

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

    template <typename NodeT, typename... Args> 
    NodeT* AddNode(const std::string& name, Args&&... args) {
        return nodes_.AddNode<NodeT>(name, std::forward<Args>(args)...);
    }

    auto Contains(const std::string& name) { return nodes_.Contains(name); }
    auto FindByName(const std::string& name) { return nodes_.FindByName(name); }

    using const_iterator = NodeContainer::const_iterator;
    const_iterator begin() const { return nodes_.begin(); }
    const_iterator end() const { return nodes_.end(); }
};

} // namespace tc

#endif // GRAPH_HPP_
