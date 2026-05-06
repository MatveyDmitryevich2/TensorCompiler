#include "gtest/gtest.h"

#include <cstring>
#include <vector>

#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "runtime/interpreter.hpp"

namespace {

std::string RawFloats(const std::vector<float>& values) {
    std::string raw(values.size() * sizeof(float), '\0');
    std::memcpy(raw.data(), values.data(), raw.size());
    return raw;
}

tc::Value* AddValue(tc::Graph* graph,
                    const std::string& name,
                    tc::Value::BelongTo belong,
                    std::vector<int64_t> shape) {
    auto* value = graph->AddNode<tc::Value>(name, belong);
    value->MergeTensorType(tc::TensorType{tc::TensorElemType::kFloat32, std::move(shape)});
    return value;
}

tc::Value* AddInitializer(tc::Graph* graph,
                          const std::string& name,
                          std::vector<int64_t> shape,
                          const std::vector<float>& data) {
    tc::TensorData tensor_data{
        tc::TensorType{tc::TensorElemType::kFloat32, std::move(shape)},
        RawFloats(data)
    };
    return graph->AddNode<tc::Value>(name, tc::Value::BelongTo::kInitializer, tensor_data);
}

} // namespace

TEST(runtime, ExecutesMatmulAddMulReluGemm) {
    tc::Graph graph;

    auto* a = AddValue(&graph, "A", tc::Value::BelongTo::kInput, {2, 3});
    auto* b = AddInitializer(&graph, "B", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto* c = AddInitializer(&graph, "C", {2, 2}, {1.0f, -100.0f, 3.0f, 4.0f});
    auto* s = AddInitializer(&graph, "S", {}, {0.5f});
    auto* gb = AddInitializer(&graph, "GB", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto* gc = AddInitializer(&graph, "GC", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* mm = AddValue(&graph, "MM", tc::Value::BelongTo::kInternal, {2, 2});
    auto* add = AddValue(&graph, "ADD", tc::Value::BelongTo::kInternal, {2, 2});
    auto* mul = AddValue(&graph, "MUL", tc::Value::BelongTo::kInternal, {2, 2});
    auto* relu = AddValue(&graph, "RELU", tc::Value::BelongTo::kInternal, {2, 2});
    auto* y = AddValue(&graph, "Y", tc::Value::BelongTo::kOutput, {2, 2});

    graph.AddNode<tc::Operation>(
        "matmul0", tc::Operation::OpType::kMatMul,
        std::vector<tc::Value*>{a, b}, std::vector<tc::Value*>{mm}
    );
    graph.AddNode<tc::Operation>(
        "add0", tc::Operation::OpType::kAdd,
        std::vector<tc::Value*>{mm, c}, std::vector<tc::Value*>{add}
    );
    graph.AddNode<tc::Operation>(
        "mul0", tc::Operation::OpType::kMul,
        std::vector<tc::Value*>{add, s}, std::vector<tc::Value*>{mul}
    );
    graph.AddNode<tc::Operation>(
        "relu0", tc::Operation::OpType::kRelu,
        std::vector<tc::Value*>{mul}, std::vector<tc::Value*>{relu}
    );
    graph.AddNode<tc::Operation>(
        "gemm0", tc::Operation::OpType::kGemm,
        std::vector<tc::Value*>{relu, gb, gc}, std::vector<tc::Value*>{y}
    );

    tc::runtime::Interpreter interpreter;
    tc::runtime::TensorMap inputs{
        {"A", tc::runtime::Tensor{{2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}}}
    };

    const tc::runtime::TensorMap outputs = interpreter.Run(graph, inputs);
    const auto& out = outputs.at("Y");

    ASSERT_EQ(out.shape, (std::vector<int64_t>{2, 2}));
    EXPECT_FLOAT_EQ(out.data[0], 12.5f);
    EXPECT_FLOAT_EQ(out.data[1], 1.0f);
    EXPECT_FLOAT_EQ(out.data[2], 27.0f);
    EXPECT_FLOAT_EQ(out.data[3], 35.0f);
}

TEST(runtime, ExecutesConvAndTranspose) {
    tc::Graph graph;

    auto* x = AddValue(&graph, "X", tc::Value::BelongTo::kInput, {1, 1, 2, 2});
    auto* w = AddInitializer(&graph, "W", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto* b = AddInitializer(&graph, "B", {1}, {1.0f});
    auto* conv = AddValue(&graph, "CONV", tc::Value::BelongTo::kInternal, {1, 1, 1, 1});
    auto* y = AddValue(&graph, "Y", tc::Value::BelongTo::kOutput, {1, 1, 1, 1});

    graph.AddNode<tc::Operation>(
        "conv0", tc::Operation::OpType::kConv,
        std::vector<tc::Value*>{x, w, b}, std::vector<tc::Value*>{conv}
    );
    graph.AddNode<tc::Operation>(
        "transpose0", tc::Operation::OpType::kTranspose,
        std::vector<tc::Value*>{conv}, std::vector<tc::Value*>{y}
    );

    tc::runtime::Interpreter interpreter;
    tc::runtime::TensorMap inputs{
        {"X", tc::runtime::Tensor{{1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}}}
    };

    const tc::runtime::TensorMap outputs = interpreter.Run(graph, inputs);
    const auto& out = outputs.at("Y");

    ASSERT_EQ(out.shape, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_FLOAT_EQ(out.data[0], 6.0f);
}
