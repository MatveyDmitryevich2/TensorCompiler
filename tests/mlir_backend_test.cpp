#include "gtest/gtest.h"

#include <cstring>
#include <string>
#include <vector>

#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "mlir_backend/mlir_backend.hpp"

namespace {

std::string RawFloat(float value) {
    std::string raw(sizeof(float), '\0');
    std::memcpy(raw.data(), &value, sizeof(float));
    return raw;
}

tc::Graph MakeMatmulMulGraph() {
    tc::Graph graph;

    auto* a = graph.AddNode<tc::Value>("A", tc::Value::BelongTo::kInput);
    a->MergeTensorType(tc::TensorType{tc::TensorElemType::kFloat32, {2, 3}});

    auto* b = graph.AddNode<tc::Value>("B", tc::Value::BelongTo::kInput);
    b->MergeTensorType(tc::TensorType{tc::TensorElemType::kFloat32, {3, 4}});

    tc::TensorData scale_data{
        tc::TensorType{tc::TensorElemType::kFloat32, {}},
        RawFloat(0.5f)
    };
    auto* scale = graph.AddNode<tc::Value>("S", tc::Value::BelongTo::kInitializer, scale_data);

    auto* mm = graph.AddNode<tc::Value>("MM", tc::Value::BelongTo::kInternal);
    mm->MergeTensorType(tc::TensorType{tc::TensorElemType::kFloat32, {2, 4}});

    auto* y = graph.AddNode<tc::Value>("Y", tc::Value::BelongTo::kOutput);
    y->MergeTensorType(tc::TensorType{tc::TensorElemType::kFloat32, {2, 4}});

    graph.AddNode<tc::Operation>(
        "matmul0",
        tc::Operation::OpType::kMatMul,
        std::vector<tc::Value*>{a, b},
        std::vector<tc::Value*>{mm}
    );
    graph.AddNode<tc::Operation>(
        "mul0",
        tc::Operation::OpType::kMul,
        std::vector<tc::Value*>{mm, scale},
        std::vector<tc::Value*>{y}
    );

    return graph;
}

} // namespace

TEST(mlir_backend, EmitsModuleForMatmulAndMul) {
    const tc::Graph graph = MakeMatmulMulGraph();

    tc::MlirBackend backend;
    const std::string mlir = backend.EmitModule(graph);

    EXPECT_NE(mlir.find("module {"), std::string::npos);
    EXPECT_NE(mlir.find("func.func @entry_main"), std::string::npos);
    EXPECT_NE(mlir.find("memref.global \"private\" constant"), std::string::npos);
    EXPECT_NE(mlir.find("memref.alloc()"), std::string::npos);
    EXPECT_NE(mlir.find("scf.for"), std::string::npos);
    EXPECT_NE(mlir.find("arith.mulf"), std::string::npos);
    EXPECT_NE(mlir.find("arith.addf"), std::string::npos);
    EXPECT_NE(mlir.find("memref.load"), std::string::npos);
    EXPECT_NE(mlir.find("memref.store"), std::string::npos);
}

TEST(mlir_backend, RejectsDynamicShapes) {
    tc::Graph graph;

    auto* x = graph.AddNode<tc::Value>("X", tc::Value::BelongTo::kInput);
    x->MergeTensorType(tc::TensorType{tc::TensorElemType::kFloat32, {-1, 4}});

    auto* y = graph.AddNode<tc::Value>("Y", tc::Value::BelongTo::kOutput);
    y->MergeTensorType(tc::TensorType{tc::TensorElemType::kFloat32, {-1, 4}});

    graph.AddNode<tc::Operation>(
        "relu0",
        tc::Operation::OpType::kRelu,
        std::vector<tc::Value*>{x},
        std::vector<tc::Value*>{y}
    );

    tc::MlirBackend backend;
    EXPECT_THROW(static_cast<void>(backend.EmitModule(graph)), std::runtime_error);
}