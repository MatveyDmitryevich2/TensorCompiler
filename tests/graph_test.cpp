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

TEST(graph, MergeTensorTypeOnlyImprovesKnownInformation) {
    Graph graph;
    Value* value = graph.AddNode<Value>("X", Value::BelongTo::kInternal);

    value->MergeTensorType(TensorType{TensorElemType::kUnknown, {}});
    ASSERT_TRUE(value->HasTensorType());
    EXPECT_EQ(value->MaybeTensorType()->ElemType(), TensorElemType::kUnknown);
    EXPECT_TRUE(value->MaybeTensorType()->Shape().empty());

    value->MergeTensorType(TensorType{TensorElemType::kFloat32, {-1, 4}});
    EXPECT_EQ(value->MaybeTensorType()->ElemType(), TensorElemType::kFloat32);
    EXPECT_EQ(value->MaybeTensorType()->Shape(), (std::vector<int64_t>{-1, 4}));

    value->MergeTensorType(TensorType{TensorElemType::kFloat32, {2, 4}});
    EXPECT_EQ(value->MaybeTensorType()->ElemType(), TensorElemType::kFloat32);
    EXPECT_EQ(value->MaybeTensorType()->Shape(), (std::vector<int64_t>{2, 4}));

    value->MergeTensorType(TensorType{TensorElemType::kUnknown, {8, 8}});
    EXPECT_EQ(value->MaybeTensorType()->ElemType(), TensorElemType::kFloat32);
    EXPECT_EQ(value->MaybeTensorType()->Shape(), (std::vector<int64_t>{2, 4}));
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

TEST(graph, ToDotHonorsOptionsAndEscapesLabels) {
    Graph graph;

    Value* x = graph.AddNode<Value>("X\"quoted", Value::BelongTo::kInput);
    Value* y = graph.AddNode<Value>("Y", Value::BelongTo::kOutput);
    AttributeMap attrs{
        {"message", Attribute{"message", std::string{"line1\nline2\""}}},
        {"perm", Attribute{"perm", std::vector<int64_t>{1, 0}}}
    };
    graph.AddNode<Operation>(
        "transpose0",
        Operation::OpType::kTranspose,
        std::vector<Value*>{x},
        std::vector<Value*>{y},
        attrs
    );

    DotOptions opt;
    opt.rank_left_to_right = true;
    const std::string dot = graph.ToDot(opt);

    EXPECT_NE(dot.find("rankdir=LR"), std::string::npos);
    EXPECT_NE(dot.find("X\\\"quoted"), std::string::npos);
    EXPECT_NE(dot.find("message="), std::string::npos);
    EXPECT_NE(dot.find("line1\\nline2"), std::string::npos);
    EXPECT_NE(dot.find("\\\""), std::string::npos);
    EXPECT_NE(dot.find("label=\"in0\""), std::string::npos);
    EXPECT_NE(dot.find("label=\"out0\""), std::string::npos);

    opt.show_attrs = false;
    opt.show_edge_indices = false;
    const std::string compact_dot = graph.ToDot(opt);
    EXPECT_EQ(compact_dot.find("message="), std::string::npos);
    EXPECT_EQ(compact_dot.find("label=\"in0\""), std::string::npos);
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
