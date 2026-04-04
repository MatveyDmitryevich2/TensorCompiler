#ifndef NODE_HPP_
#define NODE_HPP_

#include <string>
#include <cstddef>
#include <vector>
#include <cstdint>
#include <optional>
#include <utility>
#include <sstream>

#include <spdlog/spdlog.h>

#include "graph/attribute.hpp"

namespace tc {

enum class TensorElemType {
    kUnknown,
    kFloat32,
    kFloat64,
    kInt32,
    kInt64,
    kBool,
};

class TensorType {
  public:
    TensorType() = default;
    TensorType(TensorElemType elem_type, std::vector<int64_t> shape)
        : elem_type_{elem_type}, shape_{std::move(shape)} {}

    TensorElemType ElemType() const { return elem_type_; }
    const std::vector<int64_t>& Shape() const { return shape_; }

    bool HasKnownElemType() const { return elem_type_ != TensorElemType::kUnknown; }
    bool HasRank() const { return !shape_.empty(); }

    static std::string ElemTypeToStr(TensorElemType elem_type) {
        switch (elem_type) {
            case TensorElemType::kUnknown: return "unknown";
            case TensorElemType::kFloat32: return "f32";
            case TensorElemType::kFloat64: return "f64";
            case TensorElemType::kInt32:   return "i32";
            case TensorElemType::kInt64:   return "i64";
            case TensorElemType::kBool:    return "i1";
        }
        return "unknown";
    }

    std::string ToStr() const {
        std::ostringstream oss;
        oss << ElemTypeToStr(elem_type_) << "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != 0) oss << ",";
            if (shape_[i] < 0) {
                oss << "?";
            } else {
                oss << shape_[i];
            }
        }
        oss << "]";
        return oss.str();
    }

  private:
    TensorElemType elem_type_{TensorElemType::kUnknown};
    std::vector<int64_t> shape_{};
};

struct TensorData {
    TensorType type;
    std::string raw;
};

class INode {
  protected:
    std::string name_;

  public:
    INode(const std::string& name) : name_{name} {
        if (name_.empty()) { throw std::runtime_error{"INode: empty name"}; }
    }

    virtual ~INode() = default;

    virtual std::string ToStr() const = 0;
    const std::string& Name() const { return name_; }
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
      : INode{name}, belongs_{belong}, initializer_data_{std::move(data)} {
        if (initializer_data_.has_value()) {
            tensor_type_ = initializer_data_->type;
        }
      }

    ~Value() override = default;

    BelongTo GetBelongsTo() const { return belongs_; }

    void SetBelongsTo(BelongTo b) { belongs_ = b; }

    void UpgradeBelongsTo(BelongTo b) {
        if (BelongPriority(b) > BelongPriority(belongs_)) {
            belongs_ = b;
        }
    }

    bool HasTensorType() const { return tensor_type_.has_value(); }
    const std::optional<TensorType>& MaybeTensorType() const { return tensor_type_; }

    void MergeTensorType(const TensorType& type) {
        if (!tensor_type_.has_value() || !tensor_type_->HasKnownElemType()) {
            tensor_type_ = type;
            return;
        }

        if (tensor_type_->Shape().empty() && !type.Shape().empty()) {
            tensor_type_ = type;
            return;
        }

        std::vector<int64_t> merged_shape = tensor_type_->Shape();
        if (merged_shape.size() == type.Shape().size()) {
            bool improved = false;
            for (size_t i = 0; i < merged_shape.size(); ++i) {
                if (merged_shape[i] < 0 && type.Shape()[i] >= 0) {
                    merged_shape[i] = type.Shape()[i];
                    improved = true;
                }
            }
            if (improved) {
                tensor_type_ = TensorType{tensor_type_->ElemType(), std::move(merged_shape)};
            }
        }
    }

    void MergeInitializerData(std::optional<TensorData> data) {
        if (!data.has_value()) return;
        initializer_data_ = std::move(data);
        tensor_type_ = initializer_data_->type;
    }

    bool HasInitializerData() const { return initializer_data_.has_value(); }
    const std::optional<TensorData>& InitializerData() const { return initializer_data_; }

    std::string ToStr() const override { return "Value(" + Name() + ")"; }

  private:
    BelongTo belongs_;
    std::optional<TensorType> tensor_type_;
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

class Operation : public IOperation {
  public:
    enum class OpType {
        kAdd,
        kMul,
        kConv,
        kRelu,
        kMatMul,
        kGemm,
        kTranspose,
    };

  private:
    OpType op_type_;
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    AttributeMap attrs_;

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

    static std::string OpTypeToStr(OpType op) {
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

    std::string ToStr() const override {
        return OpTypeToStr(op_type_) + "(" + JoinValues(inputs_) + ") -> "
                                       + JoinValues(outputs_);
    }
};

} // namespace tc

#endif // NODE_HPP_