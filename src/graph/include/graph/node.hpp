#ifndef NODE_HPP_
#define NODE_HPP_

#include <string>
#include <cstddef>
#include <vector>
#include <cstdint>
#include <optional>
#include <utility>

#include <spdlog/spdlog.h>

#include "graph/attribute.hpp"

namespace tc {

class INode {
  private:
    std::string name_;

  public:
    INode(const std::string& name) : name_{name} {
    if (name_.empty()) { throw std::runtime_error{"INode: empty name"}; }
    }

    virtual ~INode() = default;

    virtual std::string ToStr() const = 0;
    virtual bool IsValue() const { return false; }
    const std::string& Name() const { return name_; }
};

class TensorType {/*dtype,shape*/};

struct TensorData {
    TensorType type;
    std::vector<char> raw;
};

class Value : public INode {
  public:
    enum class BelongTo {
        kInput,
        kOutput,
        kInternal,
        kInitializer,
    };

    Value(const std::string& name,
          BelongTo belong,
          std::optional<TensorData> data = std::nullopt)
      : INode{name}, belongs_{belong}, initializer_data_{std::move(data)} {}

    ~Value() override = default;

    bool IsValue() const override { return true; }

    BelongTo GetBelongsTo() const { return belongs_; }

    void SetBelongsTo(BelongTo b) { belongs_ = b; }

    void UpgradeBelongsTo(BelongTo b) {
        if (BelongPriority(b) > BelongPriority(belongs_)) {
            belongs_ = b;
        }
    }

    void MergeInitializerData(std::optional<TensorData> data) {
        if (!data.has_value()) return;
        initializer_data_ = std::move(data);
    }

    std::string ToStr() const override {
    return "Value(" + Name() + ")";
    }

  private:
    BelongTo belongs_;
    std::optional<TensorData> initializer_data_;
    
    static int BelongPriority(BelongTo b) {
        switch (b) {
            case BelongTo::kInternal:    return 0;
            case BelongTo::kInput:       return 1;
            case BelongTo::kOutput:      return 2;
            case BelongTo::kInitializer: return 3;
        }
        return 0;
    }
};

class IOperation : public INode {
  public:
    IOperation(const std::string& name) : INode{name} {}
    ~IOperation() override = default;
};

enum class OpType {
    kAdd,
    kMul,
    kConv,
    kRelu,
    kMatMul,
    kGemm,
    kTranspose,
};

class Operation : public IOperation {
  private:
    OpType op_type_;
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    AttributeMap attrs_;

    static std::string OpToStr(OpType op) {
        switch (op) {
            case OpType::kAdd:       return "Add";
            case OpType::kMul:       return "Mul";
            case OpType::kConv:      return "Conv";
            case OpType::kRelu:      return "Relu";
            case OpType::kMatMul:    return "MatMul";
            case OpType::kGemm:      return "Gemm";
            case OpType::kTranspose: return "Transpose";
        }
        return "<unknown>";
    }

    static std::string JoinValues(const std::vector<Value*>& vs) {
        std::string out;
        for (size_t i = 0; i < vs.size(); i++) {
            if (i != 0) out += ",";
            out += (vs[i] ? vs[i]->Name() : "<null>");
        }
        return out;
    }

  public:
    Operation(
        const std::string& name,
        OpType op_type,
        const std::vector<Value*>& inputs,
        const std::vector<Value*>& outputs,
        const AttributeMap& attrs = {}
    ) : IOperation{name}, op_type_{op_type}, inputs_{inputs}, outputs_{outputs}, attrs_{attrs} {}

    ~Operation() override = default;

    OpType Type() const { return op_type_; }
    const std::vector<Value*>& Inputs() const { return inputs_; }
    const std::vector<Value*>& Outputs() const { return outputs_; }
    const AttributeMap& Attrs() const { return attrs_; }

    std::string ToStr() const override {
        return OpToStr(op_type_) + "(" + JoinValues(inputs_) + ") -> "
                                       + JoinValues(outputs_);
    }
};

} // namespace tc

#endif // NODE_HPP_
