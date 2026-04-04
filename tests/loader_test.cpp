#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <string_view>

#include "onnx/onnx_pb.h"

#include "graph/attribute.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "onnx_loader/onnx_loader.hpp"

namespace fs = std::filesystem;

namespace {

onnx::ValueInfoProto* AddTensorValueInfo(onnx::GraphProto* graph,
                                         const std::string& name,
                                         int elem_type,
                                         const std::vector<int64_t>& shape,
                                         bool is_input) {
    onnx::ValueInfoProto* value = is_input ? graph->add_input() : graph->add_output();
    value->set_name(name);

    auto* tensor_type = value->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(elem_type);
    auto* tensor_shape = tensor_type->mutable_shape();
    for (int64_t dim : shape) {
        tensor_shape->add_dim()->set_dim_value(dim);
    }

    return value;
}

onnx::ValueInfoProto* AddIntermediateValueInfo(onnx::GraphProto* graph,
                                               const std::string& name,
                                               int elem_type,
                                               const std::vector<int64_t>& shape) {
    onnx::ValueInfoProto* value = graph->add_value_info();
    value->set_name(name);

    auto* tensor_type = value->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(elem_type);
    auto* tensor_shape = tensor_type->mutable_shape();
    for (int64_t dim : shape) {
        tensor_shape->add_dim()->set_dim_value(dim);
    }

    return value;
}

std::string WriteModelToTempFile(const onnx::ModelProto& model, const std::string& filename) {
    fs::path path = fs::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error{"Unable to open temp file for ONNX model"};
    }

    std::string raw;
    if (!model.SerializeToString(&raw)) {
        throw std::runtime_error{"Unable to serialize ONNX model"};
    }

    out.write(raw.data(), static_cast<std::streamsize>(raw.size()));
    out.close();
    return path.string();
}

const tc::Value& AsValue(const tc::Graph& graph, std::string_view name) {
    const std::string name_str{name};
    const tc::INode* node = graph.FindByName(name_str);
    EXPECT_NE(node, nullptr);
    const auto* value = dynamic_cast<const tc::Value*>(node);
    EXPECT_NE(value, nullptr);
    return *value;
}

const tc::Operation& AsOp(const tc::Graph& graph, std::string_view name) {
    const std::string name_str{name};
    const tc::INode* node = graph.FindByName(name_str);
    EXPECT_NE(node, nullptr);
    const auto* op = dynamic_cast<const tc::Operation*>(node);
    EXPECT_NE(op, nullptr);
    return *op;
}

} // namespace

TEST(onnx_loader, MarksInputsOutputsInitializersAndTensorTypes) {
    onnx::ModelProto model;
    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("loader_test_graph");

    AddTensorValueInfo(graph, "X", onnx::TensorProto_DataType_FLOAT, {2, 3}, true);
    AddTensorValueInfo(graph, "Y", onnx::TensorProto_DataType_FLOAT, {2, 4}, false);
    AddIntermediateValueInfo(graph, "MM_OUT", onnx::TensorProto_DataType_FLOAT, {2, 4});

    onnx::TensorProto* weights = graph->add_initializer();
    weights->set_name("W");
    weights->set_data_type(onnx::TensorProto_DataType_FLOAT);
    weights->add_dims(3);
    weights->add_dims(4);
    weights->set_raw_data(std::string(3 * 4 * static_cast<int>(sizeof(float)), '\0'));

    onnx::NodeProto* mm = graph->add_node();
    mm->set_name("matmul0");
    mm->set_op_type("MatMul");
    mm->add_input("X");
    mm->add_input("W");
    mm->add_output("MM_OUT");

    onnx::NodeProto* add = graph->add_node();
    add->set_name("add0");
    add->set_op_type("Add");
    add->add_input("MM_OUT");
    add->add_input("W");
    add->add_output("Y");
    add->add_attribute()->CopyFrom([] {
        onnx::AttributeProto attr;
        attr.set_name("broadcast_hint");
        attr.set_type(onnx::AttributeProto::INT);
        attr.set_i(1);
        return attr;
    }());

    const std::string model_path = WriteModelToTempFile(model, "tc_loader_test_basic.onnx");

    tc::OnnxLoader loader;
    tc::Graph loaded = loader.Load(model_path);

    const tc::Value& x = AsValue(loaded, "X");
    EXPECT_EQ(x.GetBelongsTo(), tc::Value::BelongTo::kInput);
    ASSERT_TRUE(x.HasTensorType());
    EXPECT_EQ(x.MaybeTensorType()->ElemType(), tc::TensorElemType::kFloat32);
    EXPECT_EQ(x.MaybeTensorType()->Shape(), (std::vector<int64_t>{2, 3}));

    const tc::Value& y = AsValue(loaded, "Y");
    EXPECT_EQ(y.GetBelongsTo(), tc::Value::BelongTo::kOutput);
    ASSERT_TRUE(y.HasTensorType());
    EXPECT_EQ(y.MaybeTensorType()->Shape(), (std::vector<int64_t>{2, 4}));

    const tc::Value& w = AsValue(loaded, "W");
    EXPECT_EQ(w.GetBelongsTo(), tc::Value::BelongTo::kInitializer);
    ASSERT_TRUE(w.HasTensorType());
    ASSERT_TRUE(w.HasInitializerData());
    EXPECT_EQ(w.MaybeTensorType()->Shape(), (std::vector<int64_t>{3, 4}));
    EXPECT_EQ(w.InitializerData()->raw.size(), static_cast<size_t>(3 * 4 * sizeof(float)));

    const tc::Value& mm_out = AsValue(loaded, "MM_OUT");
    EXPECT_EQ(mm_out.GetBelongsTo(), tc::Value::BelongTo::kInternal);
    ASSERT_TRUE(mm_out.HasTensorType());
    EXPECT_EQ(mm_out.MaybeTensorType()->Shape(), (std::vector<int64_t>{2, 4}));

    const tc::Operation& matmul = AsOp(loaded, "matmul0");
    EXPECT_EQ(matmul.Type(), tc::Operation::OpType::kMatMul);
    ASSERT_EQ(matmul.Inputs().size(), 2U);
    ASSERT_EQ(matmul.Outputs().size(), 1U);
    EXPECT_EQ(matmul.Inputs()[0]->Name(), "X");
    EXPECT_EQ(matmul.Inputs()[1]->Name(), "W");
    EXPECT_EQ(matmul.Outputs()[0]->Name(), "MM_OUT");

    const tc::Operation& add_op = AsOp(loaded, "add0");
    EXPECT_EQ(add_op.Type(), tc::Operation::OpType::kAdd);
    ASSERT_TRUE(add_op.Attrs().contains("broadcast_hint"));
    EXPECT_EQ(add_op.Attrs().at("broadcast_hint").As<int64_t>(), 1);

    fs::remove(model_path);
}

TEST(onnx_loader, ParsesVectorAttributesForConv) {
    onnx::ModelProto model;
    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("loader_test_conv_graph");

    AddTensorValueInfo(graph, "X", onnx::TensorProto_DataType_FLOAT, {1, 3, 8, 8}, true);
    AddTensorValueInfo(graph, "Y", onnx::TensorProto_DataType_FLOAT, {1, 4, 8, 8}, false);

    onnx::TensorProto* weights = graph->add_initializer();
    weights->set_name("W_conv");
    weights->set_data_type(onnx::TensorProto_DataType_FLOAT);
    weights->add_dims(4);
    weights->add_dims(3);
    weights->add_dims(3);
    weights->add_dims(3);
    weights->set_raw_data(std::string(4 * 3 * 3 * 3 * static_cast<int>(sizeof(float)), '\0'));

    onnx::TensorProto* bias = graph->add_initializer();
    bias->set_name("B_conv");
    bias->set_data_type(onnx::TensorProto_DataType_FLOAT);
    bias->add_dims(4);
    bias->set_raw_data(std::string(4 * static_cast<int>(sizeof(float)), '\0'));

    onnx::NodeProto* conv = graph->add_node();
    conv->set_name("conv0");
    conv->set_op_type("Conv");
    conv->add_input("X");
    conv->add_input("W_conv");
    conv->add_input("B_conv");
    conv->add_output("Y");

    auto* strides = conv->add_attribute();
    strides->set_name("strides");
    strides->set_type(onnx::AttributeProto::INTS);
    strides->add_ints(1);
    strides->add_ints(1);

    auto* dilations = conv->add_attribute();
    dilations->set_name("dilations");
    dilations->set_type(onnx::AttributeProto::INTS);
    dilations->add_ints(2);
    dilations->add_ints(2);

    auto* group = conv->add_attribute();
    group->set_name("group");
    group->set_type(onnx::AttributeProto::INT);
    group->set_i(1);

    const std::string model_path = WriteModelToTempFile(model, "tc_loader_test_conv.onnx");

    tc::OnnxLoader loader;
    tc::Graph loaded = loader.Load(model_path);

    const tc::Operation& op = AsOp(loaded, "conv0");
    EXPECT_EQ(op.Type(), tc::Operation::OpType::kConv);
    ASSERT_TRUE(op.Attrs().contains("strides"));
    ASSERT_TRUE(op.Attrs().contains("dilations"));
    ASSERT_TRUE(op.Attrs().contains("group"));
    EXPECT_EQ(op.Attrs().at("strides").As<std::vector<int64_t>>(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(op.Attrs().at("dilations").As<std::vector<int64_t>>(), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(op.Attrs().at("group").As<int64_t>(), 1);

    const tc::Value& y = AsValue(loaded, "Y");
    EXPECT_EQ(y.GetBelongsTo(), tc::Value::BelongTo::kOutput);
    ASSERT_TRUE(y.HasTensorType());
    EXPECT_EQ(y.MaybeTensorType()->Shape(), (std::vector<int64_t>{1, 4, 8, 8}));

    fs::remove(model_path);
}