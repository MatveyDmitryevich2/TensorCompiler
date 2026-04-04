#include "gtest/gtest.h"

#include <stdexcept>
#include <string>
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