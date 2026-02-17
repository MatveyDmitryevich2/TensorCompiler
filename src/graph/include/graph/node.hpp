#ifndef NODE_HPP_
#define NODE_HPP_

#include <filesystem>
#include <string>

#include <spdlog/spdlog.h>

namespace tc {

class INode {
  private:
    std::string name_;
  public:
    INode(const std::string& name) : name_{name != "" ? name : "<no name>"} {}
    virtual ~INode() = default;

    virtual std::string ToStr() const = 0;
    std::string Name() const { return name_; }
};

class Type {};
class Attribute {};

class Value : public INode {
  public:
    enum class BelongTo {
        kInput,
        kOutput,
        kInternal,
    };
  private:
    BelongTo belongs_;
  public:
    Value(const std::string& name, BelongTo belong) : INode{name}, belongs_{belong} {}
    ~Value() override = default;
    std::string ToStr() const override { return Name(); }
};

class IOperation : public INode {
  public:
    IOperation(const std::string& name) : INode{name} {}
    ~IOperation() override = default;
};

// FIXME: add other nodes
// FIXME: support attributes and types 

// A + B = C
class Add : public IOperation {
  private:
    Value* a_;
    Value* b_;
    Value* c_;
  public:
    Add(
        const std::string& name, 
        Value* A, 
        Value* B, 
        Value* C
    ) : IOperation{name}, a_{A}, b_{B}, c_{C} {}

    ~Add() override = default;
    std::string ToStr() const override { 
        return "Add(" + a_->Name() + "," + b_->Name() + ") -> " + c_->Name(); 
    }
};

// A * B = Y
class MatMul : public IOperation {
  private:
    Value* a_;
    Value* b_;
    Value* y_;
  public:
    MatMul(
        const std::string& name, 
        Value* A, 
        Value* B, 
        Value* Y
    ) : IOperation{name}, a_{A}, b_{B}, y_{Y} {}

    ~MatMul() override = default;
    std::string ToStr() const override { 
        return "MatMul(" + a_->Name() + "," + b_->Name() + ") -> " + y_->Name(); 
    }
};

class Transpose : public IOperation {
  private:
    Value* data_;
    Value* transposed_;
  public:
    Transpose(
        const std::string& name, 
        Value* data, 
        Value* transposed
    ) : IOperation{name}, data_{data}, transposed_{transposed} {}

    ~Transpose() override = default;
    std::string ToStr() const override { 
        return "Transpose(" + data_->Name() + ") -> " + transposed_->Name(); 
    }
};

class Mul : public IOperation {
    Mul(const std::string& name) : IOperation{name} {}
    ~Mul() override = default;
    std::string ToStr() const override { return "Mul " + Name() + "\n"; }
};
class Conv : public IOperation {
    Conv(const std::string& name) : IOperation{name} {}
    ~Conv() override = default;
    std::string ToStr() const override { return "Conv " + Name() + "\n"; }
};
class Relu : public IOperation {
    Relu(const std::string& name) : IOperation{name} {}
    ~Relu() override = default;
    std::string ToStr() const override { return "Relu " + Name() + "\n"; }
};
class Gemm : public IOperation {
    Gemm(const std::string& name) : IOperation{name} {}
    ~Gemm() override = default;
    std::string ToStr() const override { return "Gemm " + Name() + "\n"; }
};

} // namespace tc

#endif // NODE_HPP_
