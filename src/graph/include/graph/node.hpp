#ifndef NODE_HPP_
#define NODE_HPP_

#include <filesystem>
#include <string>

namespace tc {

class INode {
  public:
    virtual ~INode() = default;
    virtual std::string to_str() = 0;
};

class InputNode : public INode {
  public:
    ~InputNode() override = default;
    std::string to_str() override { return "Input"; }
};

class OutputNode : public INode {
  public:
    ~OutputNode() override = default;
    std::string to_str() override { return "Output"; }
};

class IOperations : public INode {
  public:
    ~IOperations() override = default;
    std::string to_str() override = 0;
};

// FIXME: add other nodes
// FIXME: support attributes and types 

// A + B = C
class Add : public IOperations {
  private:
    InputNode* a_;
    InputNode* b_;
    OutputNode* c_;
  public:
    Add(InputNode* A, InputNode* B, OutputNode* C) : a_{A}, b_{B}, c_{C} {}
    ~Add() override = default;
    std::string to_str() override { return "Add(a,b) -> c"; }
};

// A * B = Y
class MatMul : public INode {
  public:
    InputNode* a_;
    InputNode* b_;
    OutputNode* y_;
  public:
    MatMul(InputNode* A, InputNode* B, OutputNode* Y) : a_{A}, b_{B}, y_{Y} {}
    ~MatMul() override = default;
    std::string to_str() override { return "MatMul(a,b) -> y"; }
};

} // namespace tc

#endif // NODE_HPP_
