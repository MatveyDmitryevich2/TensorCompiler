#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph/graph.hpp"
#include "graph/node.hpp"

using namespace tc;

TEST(graph, AddNode) {
    Graph graph;

    Value* val = graph.AddNode<Value>("X", Value::BelongTo::kInternal);
    ASSERT_NE(val, nullptr);

    EXPECT_TRUE(graph.Contains("X"));

    INode* found = graph.FindByName("X");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(found, val);
}

TEST(graph, FindForMissing) {
    Graph graph;

    EXPECT_FALSE(graph.Contains("missing"));
    EXPECT_EQ(graph.FindByName("missing"), nullptr);
}

TEST(graph, ReusesValueAndUpgradesBelonging) {
    Graph graph;

    Value* value = graph.AddNode<Value>("W", Value::BelongTo::kInternal);
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->GetBelongsTo(), Value::BelongTo::kInternal);

    std::optional<TensorData> data = TensorData{
        TensorType{TensorElemType::kFloat32, {3, 4}},
        std::string(3 * 4 * static_cast<int>(sizeof(float)), '\0')
    };

    Value* same = graph.AddNode<Value>("W", Value::BelongTo::kInitializer, std::move(data));
    ASSERT_EQ(same, value);
    EXPECT_EQ(same->GetBelongsTo(), Value::BelongTo::kInitializer);
    ASSERT_TRUE(same->HasTensorType());
    ASSERT_TRUE(same->HasInitializerData());
    EXPECT_EQ(same->MaybeTensorType()->ElemType(), TensorElemType::kFloat32);
    EXPECT_EQ(same->MaybeTensorType()->Shape(), (std::vector<int64_t>{3, 4}));
}

TEST(graph, FindsTypedNodes) {
    Graph graph;

    Value* x = graph.AddNode<Value>("X", Value::BelongTo::kInput);
    Value* y = graph.AddNode<Value>("Y", Value::BelongTo::kOutput);
    Operation* op = graph.AddNode<Operation>(
        "relu0",
        Operation::OpType::kRelu,
        std::vector<Value*>{x},
        std::vector<Value*>{y}
    );

    EXPECT_EQ(graph.FindValueByName("X"), x);
    EXPECT_EQ(graph.FindOperationByName("relu0"), op);
    EXPECT_EQ(graph.FindValueByName("relu0"), nullptr);
    EXPECT_EQ(graph.FindOperationByName("X"), nullptr);

    EXPECT_EQ(graph.Values().size(), 2);
    EXPECT_EQ(graph.ValuesByBelong(Value::BelongTo::kInput).size(), 1);
    EXPECT_EQ(graph.ValuesByBelong(Value::BelongTo::kOutput).size(), 1);
    EXPECT_EQ(graph.Operations().size(), 1);
}

TEST(graph, RejectsDuplicateOperationNames) {
    Graph graph;

    Value* x = graph.AddNode<Value>("X", Value::BelongTo::kInput);
    Value* y = graph.AddNode<Value>("Y", Value::BelongTo::kOutput);
    graph.AddNode<Operation>(
        "relu0",
        Operation::OpType::kRelu,
        std::vector<Value*>{x},
        std::vector<Value*>{y}
    );

    EXPECT_THROW(
        graph.AddNode<Operation>(
            "relu0",
            Operation::OpType::kRelu,
            std::vector<Value*>{x},
            std::vector<Value*>{y}
        ),
        std::runtime_error
    );
    EXPECT_THROW(graph.AddNode<Value>("relu0", Value::BelongTo::kInternal), std::runtime_error);
}

TEST(graph, MovesNodeOwnership) {
    Graph graph;
    graph.AddNode<Value>("X", Value::BelongTo::kInput);

    Graph moved{std::move(graph)};

    EXPECT_NE(moved.FindValueByName("X"), nullptr);
    EXPECT_EQ(moved.Values().size(), 1);
}
