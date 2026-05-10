#include "gtest/gtest.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph/attribute.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "runtime/interpreter.hpp"
#include "runtime/tensor.hpp"

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

TEST(runtime, BroadcastsElementwiseInputsAndUsesExplicitTransposePerm) {
    tc::Graph graph;

    auto* x = AddValue(&graph, "X", tc::Value::BelongTo::kInput, {2, 1, 3});
    auto* row = AddInitializer(&graph, "ROW", {3}, {10.0f, 20.0f, 30.0f});
    auto* add = AddValue(&graph, "ADD", tc::Value::BelongTo::kInternal, {2, 1, 3});
    auto* y = AddValue(&graph, "Y", tc::Value::BelongTo::kOutput, {3, 2, 1});

    graph.AddNode<tc::Operation>(
        "add0", tc::Operation::OpType::kAdd,
        std::vector<tc::Value*>{x, row}, std::vector<tc::Value*>{add}
    );
    graph.AddNode<tc::Operation>(
        "transpose0", tc::Operation::OpType::kTranspose,
        std::vector<tc::Value*>{add}, std::vector<tc::Value*>{y},
        tc::AttributeMap{{"perm", tc::Attribute{"perm", std::vector<int64_t>{2, 0, 1}}}}
    );

    tc::runtime::Interpreter interpreter;
    const tc::runtime::TensorMap outputs = interpreter.Run(
        graph,
        {{"X", tc::runtime::Tensor{{2, 1, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}}}}
    );
    const auto& out = outputs.at("Y");

    ASSERT_EQ(out.shape, (std::vector<int64_t>{3, 2, 1}));
    ASSERT_EQ(out.data.size(), 6U);
    EXPECT_FLOAT_EQ(out.data[0], 11.0f);
    EXPECT_FLOAT_EQ(out.data[1], 14.0f);
    EXPECT_FLOAT_EQ(out.data[2], 22.0f);
    EXPECT_FLOAT_EQ(out.data[3], 25.0f);
    EXPECT_FLOAT_EQ(out.data[4], 33.0f);
    EXPECT_FLOAT_EQ(out.data[5], 36.0f);
}

TEST(runtime, ExecutesGemmWithTransposeScalingAndBroadcastBias) {
    tc::Graph graph;

    auto* a = AddValue(&graph, "A", tc::Value::BelongTo::kInput, {3, 2});
    auto* b = AddInitializer(&graph, "B", {3, 2}, {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f});
    auto* c = AddInitializer(&graph, "C", {2}, {1.0f, -1.0f});
    auto* y = AddValue(&graph, "Y", tc::Value::BelongTo::kOutput, {2, 2});

    graph.AddNode<tc::Operation>(
        "gemm0", tc::Operation::OpType::kGemm,
        std::vector<tc::Value*>{a, b, c}, std::vector<tc::Value*>{y},
        tc::AttributeMap{
            {"transA", tc::Attribute{"transA", int64_t{1}}},
            {"alpha", tc::Attribute{"alpha", 0.5f}},
            {"beta", tc::Attribute{"beta", 2.0f}}
        }
    );

    tc::runtime::Interpreter interpreter;
    const tc::runtime::TensorMap outputs = interpreter.Run(
        graph,
        {{"A", tc::runtime::Tensor{{3, 2}, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}}}}
    );
    const auto& out = outputs.at("Y");

    ASSERT_EQ(out.shape, (std::vector<int64_t>{2, 2}));
    ASSERT_EQ(out.data.size(), 4U);
    EXPECT_FLOAT_EQ(out.data[0], 9.0f);
    EXPECT_FLOAT_EQ(out.data[1], 68.0f);
    EXPECT_FLOAT_EQ(out.data[2], 18.0f);
    EXPECT_FLOAT_EQ(out.data[3], 158.0f);
}

TEST(runtime, RejectsInvalidInputsAndInitializers) {
    tc::Graph graph;
    auto* x = AddValue(&graph, "X", tc::Value::BelongTo::kInput, {2});
    auto* y = AddValue(&graph, "Y", tc::Value::BelongTo::kOutput, {2});
    graph.AddNode<tc::Operation>(
        "relu0", tc::Operation::OpType::kRelu,
        std::vector<tc::Value*>{x}, std::vector<tc::Value*>{y}
    );

    tc::runtime::Interpreter interpreter;
    EXPECT_THROW(static_cast<void>(interpreter.Run(graph, {})), std::runtime_error);
    EXPECT_THROW(
        static_cast<void>(interpreter.Run(graph, {{"X", tc::runtime::Tensor{{3}, {1.0f, 2.0f, 3.0f}}}})),
        std::runtime_error
    );
    EXPECT_THROW(
        static_cast<void>(interpreter.Run(
            graph,
            {
                {"X", tc::runtime::Tensor{{2}, {1.0f, 2.0f}}},
                {"EXTRA", tc::runtime::Tensor{{1}, {0.0f}}}
            }
        )),
        std::runtime_error
    );

    tc::Graph bad_init_graph;
    auto* bad = bad_init_graph.AddNode<tc::Value>(
        "BAD",
        tc::Value::BelongTo::kInitializer,
        tc::TensorData{tc::TensorType{tc::TensorElemType::kFloat32, {2}}, std::string(sizeof(float), '\0')}
    );
    auto* bad_y = AddValue(&bad_init_graph, "Y", tc::Value::BelongTo::kOutput, {2});
    bad_init_graph.AddNode<tc::Operation>(
        "relu_bad", tc::Operation::OpType::kRelu,
        std::vector<tc::Value*>{bad}, std::vector<tc::Value*>{bad_y}
    );
    EXPECT_THROW(static_cast<void>(interpreter.Run(bad_init_graph, {})), std::runtime_error);
}

TEST(runtime_tensor, FormatsShapesAndCountsElements) {
    EXPECT_EQ(tc::runtime::ShapeToStr({2, 3, 4}), "[2,3,4]");
    EXPECT_EQ(tc::runtime::ShapeToStr({}), "[]");
    EXPECT_EQ((tc::runtime::Tensor{{2, 3, 4}, {}}).NumElements(), 24U);
    EXPECT_THROW(static_cast<void>((tc::runtime::Tensor{{2, -1}, {}}).NumElements()), std::runtime_error);
}
