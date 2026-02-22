#include "gtest/gtest.h"

#include <stdexcept>
#include <string>

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

