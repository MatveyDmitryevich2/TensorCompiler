#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <memory>
#include <utility>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "graph/node.hpp"

namespace tc {

class NodeContainer {
  private:
    using NodesOwner = std::vector<INode*>;
    using NameTable = std::unordered_map<std::string, INode*>;

    NodesOwner nodes_;
    NameTable name_table_;

    static void MergeExistingValue(Value* value, Value::BelongTo belong) {
        value->UpgradeBelongsTo(belong);
    }

    static void MergeExistingValue(Value* value,
                                   Value::BelongTo belong,
                                   std::optional<TensorData> data) {
        value->UpgradeBelongsTo(belong);
        value->MergeInitializerData(std::move(data));
    }

    void Clear() {
        for (INode* node : nodes_) {
            delete node;
        }
        nodes_.clear();
        name_table_.clear();
    }

  public:
    using const_iterator = std::vector<INode*>::const_iterator;

    NodeContainer() = default;
    NodeContainer(const NodeContainer& other) = delete;
    NodeContainer& operator=(const NodeContainer& other) = delete;

    NodeContainer(NodeContainer&& other)
        : nodes_{std::move(other.nodes_)}, name_table_{std::move(other.name_table_)} {
        other.nodes_.clear();
        other.name_table_.clear();
    }

    NodeContainer& operator=(NodeContainer&& other) {
        if (this != &other) {
            Clear();
            nodes_ = std::move(other.nodes_);
            name_table_ = std::move(other.name_table_);
            other.nodes_.clear();
            other.name_table_.clear();
        }
        return *this;
    }

    ~NodeContainer() {
        Clear();
    }

    template <typename NodeT, typename... Args>
    NodeT* AddNode(const std::string& name, Args&&... args) {
        static_assert(std::is_base_of_v<INode, NodeT>, "NodeT should be derived from INode");

        auto node_it = name_table_.find(name);
        if (node_it != name_table_.end()) {
            if constexpr (std::is_same_v<NodeT, Value>) {
                if (node_it->second->Kind() != NodeKind::kValue) {
                    throw std::runtime_error{"Node name already belongs to an operation: " + name};
                }

                Value* value = static_cast<Value*>(node_it->second);
                MergeExistingValue(value, std::forward<Args>(args)...);
                return value;
            } else {
                throw std::runtime_error{"Duplicate graph node name: " + name};
            }
        }

        std::unique_ptr<NodeT> node = std::make_unique<NodeT>(name, std::forward<Args>(args)...);
        NodeT* raw_ptr = node.get();

        nodes_.push_back(raw_ptr);
        try {
            const bool inserted = name_table_.emplace(name, raw_ptr).second;
            if (!inserted) {
                throw std::runtime_error{"Duplicate graph node name: " + name};
            }
        } catch (...) {
            nodes_.pop_back();
            throw;
        }
        node.release();

        return raw_ptr;
    }

    bool Contains(const std::string& name) const {
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

    const INode* FindByName(const std::string& name) const {
        auto&& node = name_table_.find(name);
        if (node == name_table_.end()) {
            SPDLOG_TRACE("Not found {}", name);
            return nullptr;
        }

        return node->second;
    }

    Value* FindValueByName(const std::string& name) {
        INode* node = FindByName(name);
        if (node == nullptr || node->Kind() != NodeKind::kValue) {
            return nullptr;
        }
        return static_cast<Value*>(node);
    }

    const Value* FindValueByName(const std::string& name) const {
        const INode* node = FindByName(name);
        if (node == nullptr || node->Kind() != NodeKind::kValue) {
            return nullptr;
        }
        return static_cast<const Value*>(node);
    }

    const Operation* FindOperationByName(const std::string& name) const {
        const INode* node = FindByName(name);
        if (node == nullptr || node->Kind() != NodeKind::kOperation) {
            return nullptr;
        }
        return static_cast<const Operation*>(node);
    }

    std::vector<const Value*> Values() const {
        std::vector<const Value*> values;
        values.reserve(nodes_.size());
        for (const INode* node : nodes_) {
            if (node->Kind() == NodeKind::kValue) {
                values.push_back(static_cast<const Value*>(node));
            }
        }
        return values;
    }

    std::vector<const Value*> ValuesByBelong(Value::BelongTo belong) const {
        std::vector<const Value*> values;
        values.reserve(nodes_.size());
        for (const INode* node : nodes_) {
            if (node->Kind() == NodeKind::kValue) {
                const Value* value = static_cast<const Value*>(node);
                if (value->GetBelongsTo() == belong) {
                    values.push_back(value);
                }
            }
        }
        return values;
    }

    std::vector<const Operation*> Operations() const {
        std::vector<const Operation*> operations;
        operations.reserve(nodes_.size());
        for (const INode* node : nodes_) {
            if (node->Kind() == NodeKind::kOperation) {
                operations.push_back(static_cast<const Operation*>(node));
            }
        }
        return operations;
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

struct DotOptions {
    bool show_values = true;
    bool show_attrs = true;
    bool show_edge_indices = true;
    size_t max_attr_chars = 140;
    size_t max_attr_items = 16;
    bool rank_left_to_right = false;
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

    bool Contains(const std::string& name) const { return nodes_.Contains(name); }
    INode* FindByName(const std::string& name) { return nodes_.FindByName(name); }
    const INode* FindByName(const std::string& name) const { return nodes_.FindByName(name); }
    Value* FindValueByName(const std::string& name) { return nodes_.FindValueByName(name); }
    const Value* FindValueByName(const std::string& name) const { return nodes_.FindValueByName(name); }
    const Operation* FindOperationByName(const std::string& name) const { return nodes_.FindOperationByName(name); }
    std::vector<const Value*> Values() const { return nodes_.Values(); }
    std::vector<const Value*> ValuesByBelong(Value::BelongTo belong) const { return nodes_.ValuesByBelong(belong); }
    std::vector<const Operation*> Operations() const { return nodes_.Operations(); }
    std::string ToDot(const DotOptions& opt = {}) const;

    using const_iterator = NodeContainer::const_iterator;
    const_iterator begin() const { return nodes_.begin(); }
    const_iterator end() const { return nodes_.end(); }
};

} // namespace tc

#endif // GRAPH_HPP_
