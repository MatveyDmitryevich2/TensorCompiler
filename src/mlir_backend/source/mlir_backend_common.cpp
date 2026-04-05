#include "mlir_backend_internal.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace tc::detail {

void Fail(const std::string& message) {
    throw std::runtime_error{"MLIR backend: " + message};
}

bool IsFloatType(TensorElemType elem_type) {
    return elem_type == TensorElemType::kFloat32 || elem_type == TensorElemType::kFloat64;
}

bool IsIntegerLikeType(TensorElemType elem_type) {
    return elem_type == TensorElemType::kInt32 || elem_type == TensorElemType::kInt64 || elem_type == TensorElemType::kBool;
}

std::string ElemTypeToMlir(TensorElemType elem_type) {
    switch (elem_type) {
        case TensorElemType::kFloat32: return "f32";
        case TensorElemType::kFloat64: return "f64";
        case TensorElemType::kInt32: return "i32";
        case TensorElemType::kInt64: return "i64";
        case TensorElemType::kBool: return "i1";
        case TensorElemType::kUnknown: break;
    }
    Fail("unknown tensor element type");
}

namespace {

std::string ShapePrefixToMlir(const std::vector<int64_t>& shape) {
    std::string out;
    for (int64_t dim : shape) {
        if (dim < 0) {
            Fail("dynamic shapes are not supported by the MLIR emitter");
        }
        out += std::to_string(dim);
        out += "x";
    }
    return out;
}

int64_t NumElements(const std::vector<int64_t>& shape) {
    int64_t total = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            Fail("dynamic shapes are not supported by the MLIR emitter");
        }
        total *= dim;
    }
    return total;
}

template <typename T>
std::vector<T> ReadPodValues(const std::string& raw, size_t count) {
    const size_t expected_bytes = count * sizeof(T);
    if (raw.size() != expected_bytes) {
        Fail("initializer raw byte size mismatch");
    }

    std::vector<T> out(count);
    if (expected_bytes != 0) {
        std::memcpy(out.data(), raw.data(), expected_bytes);
    }
    return out;
}

std::string FormatFloat(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }

    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    std::string out = oss.str();
    if (out == "-0") {
        out = "0";
    }
    if (out.find_first_of(".eE") == std::string::npos) {
        out += ".0";
    }
    if (out == "-0.0") {
        out = "0.0";
    }
    return out;
}

template <typename T, typename Formatter>
std::string DenseRecursive(const std::vector<T>& values,
                           const std::vector<int64_t>& shape,
                           size_t dim,
                           size_t& pos,
                           Formatter formatter) {
    if (dim == shape.size()) {
        if (pos >= values.size()) {
            Fail("initializer traversal overflow");
        }
        return formatter(values[pos++]);
    }

    std::string out = "[";
    for (int64_t i = 0; i < shape[dim]; ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += DenseRecursive(values, shape, dim + 1, pos, formatter);
    }
    out += "]";
    return out;
}

} // namespace

std::string MemRefTypeToMlir(const TensorType& type) {
    if (!type.HasKnownElemType()) {
        Fail("tensor type without known element type");
    }
    return "memref<" + ShapePrefixToMlir(type.Shape()) + ElemTypeToMlir(type.ElemType()) + ">";
}

std::string DenseLiteral(const TensorData& data) {
    const std::vector<int64_t>& shape = data.type.Shape();
    const size_t count = static_cast<size_t>(NumElements(shape));

    auto make_dense = [&](const auto& values, auto formatter) -> std::string {
        if (shape.empty()) {
            if (values.empty()) {
                Fail("scalar initializer has no payload");
            }
            return "dense<" + formatter(values[0]) + ">";
        }
        size_t pos = 0;
        const std::string body = DenseRecursive(values, shape, 0, pos, formatter);
        if (pos != values.size()) {
            Fail("initializer traversal underflow/overflow");
        }
        return "dense<" + body + ">";
    };

    switch (data.type.ElemType()) {
        case TensorElemType::kFloat32: {
            const auto values = ReadPodValues<float>(data.raw, count);
            return make_dense(values, [](float v) { return FormatFloat(static_cast<double>(v)); });
        }
        case TensorElemType::kFloat64: {
            const auto values = ReadPodValues<double>(data.raw, count);
            return make_dense(values, [](double v) { return FormatFloat(v); });
        }
        case TensorElemType::kInt32: {
            const auto values = ReadPodValues<int32_t>(data.raw, count);
            return make_dense(values, [](int32_t v) { return std::to_string(v); });
        }
        case TensorElemType::kInt64: {
            const auto values = ReadPodValues<int64_t>(data.raw, count);
            return make_dense(values, [](int64_t v) { return std::to_string(v); });
        }
        case TensorElemType::kBool: {
            const auto values = ReadPodValues<uint8_t>(data.raw, count);
            return make_dense(values, [](uint8_t v) { return v == 0 ? std::string{"false"} : std::string{"true"}; });
        }
        case TensorElemType::kUnknown:
            break;
    }

    Fail("unsupported initializer element type");
}

std::string SanitizeIdentifier(std::string_view value, std::string_view prefix) {
    std::string out;
    out.reserve(value.size() + prefix.size() + 1);

    auto is_ident_char = [](char c) {
        return (c >= 'a' && c <= 'z') ||
               (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') ||
               c == '_' || c == '$';
    };

    for (char c : value) {
        out += is_ident_char(c) ? c : '_';
    }

    if (out.empty() || !((out[0] >= 'a' && out[0] <= 'z') || (out[0] >= 'A' && out[0] <= 'Z') || out[0] == '_')) {
        out.insert(out.begin(), '_');
    }

    if (!prefix.empty()) {
        out = std::string(prefix) + "_" + out;
    }
    return out;
}

std::vector<const Value*> CollectValuesByBelong(const Graph& graph, Value::BelongTo belong) {
    std::vector<const Value*> values;
    for (const INode* node : graph) {
        const auto* value = dynamic_cast<const Value*>(node);
        if (value != nullptr && value->GetBelongsTo() == belong) {
            values.push_back(value);
        }
    }
    return values;
}

std::vector<const Value*> CollectInternalValues(const Graph& graph) {
    std::vector<const Value*> values;
    for (const INode* node : graph) {
        const auto* value = dynamic_cast<const Value*>(node);
        if (value != nullptr && value->GetBelongsTo() == Value::BelongTo::kInternal) {
            values.push_back(value);
        }
    }
    return values;
}

std::vector<const Operation*> CollectOperations(const Graph& graph) {
    std::vector<const Operation*> ops;
    for (const INode* node : graph) {
        const auto* op = dynamic_cast<const Operation*>(node);
        if (op != nullptr) {
            ops.push_back(op);
        }
    }
    return ops;
}

const TensorType& RequireTensorType(const Value& value) {
    if (!value.HasTensorType()) {
        Fail("value '" + value.Name() + "' has no tensor type");
    }
    return *value.MaybeTensorType();
}

float GetFloatAttr(const AttributeMap& attrs, const std::string& name, float default_value) {
    auto it = attrs.find(name);
    if (it == attrs.end()) {
        return default_value;
    }
    return it->second.As<float>();
}

int64_t GetIntAttr(const AttributeMap& attrs, const std::string& name, int64_t default_value) {
    auto it = attrs.find(name);
    if (it == attrs.end()) {
        return default_value;
    }
    return it->second.As<int64_t>();
}

std::vector<int64_t> GetIntsAttr(const AttributeMap& attrs,
                                 const std::string& name,
                                 const std::vector<int64_t>& default_value) {
    auto it = attrs.find(name);
    if (it == attrs.end()) {
        return default_value;
    }
    return it->second.As<std::vector<int64_t>>();
}

std::string ModuleEmitter::EmitIndexConst(int64_t value) {
    const std::string name = NewSsa("cidx");
    EmitLine(name + " = arith.constant " + std::to_string(value) + " : index");
    return name;
}

std::string ModuleEmitter::EmitNumericConst(TensorElemType elem_type, double value) {
    const std::string name = NewSsa("cst");
    if (elem_type == TensorElemType::kBool) {
        EmitLine(name + " = arith.constant " + (value == 0.0 ? std::string{"false"} : std::string{"true"}) + " : i1");
        return name;
    }
    if (IsFloatType(elem_type)) {
        EmitLine(name + " = arith.constant " + FormatFloat(value) + " : " + ElemTypeToMlir(elem_type));
        return name;
    }
    EmitLine(name + " = arith.constant " + std::to_string(static_cast<int64_t>(value)) + " : " + ElemTypeToMlir(elem_type));
    return name;
}

std::string ModuleEmitter::EmitLoadRaw(const std::string& memref,
                                       const std::string& memref_type,
                                       const std::vector<std::string>& indices,
                                       std::string_view hint) {
    const std::string name = NewSsa(hint);
    std::string index_list;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) {
            index_list += ", ";
        }
        index_list += indices[i];
    }
    EmitLine(name + " = memref.load " + memref + "[" + index_list + "] : " + memref_type);
    return name;
}

void ModuleEmitter::EmitStoreRaw(const std::string& scalar,
                                 const std::string& memref,
                                 const std::string& memref_type,
                                 const std::vector<std::string>& indices) {
    std::string index_list;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) {
            index_list += ", ";
        }
        index_list += indices[i];
    }
    EmitLine("memref.store " + scalar + ", " + memref + "[" + index_list + "] : " + memref_type);
}

std::string ModuleEmitter::EmitLoadValue(const Value& value,
                                         const std::vector<std::string>& indices,
                                         std::string_view hint) {
    return EmitLoadRaw(RefOf(value), MemRefType(value), indices, hint);
}

void ModuleEmitter::EmitStoreValue(const std::string& scalar,
                                   const Value& value,
                                   const std::vector<std::string>& indices) {
    EmitStoreRaw(scalar, RefOf(value), MemRefType(value), indices);
}

std::vector<std::string> ModuleEmitter::BroadcastIndices(const Value& src,
                                                         const Value& dst,
                                                         const std::vector<std::string>& dst_indices) {
    const std::vector<int64_t>& src_shape = ShapeOf(src);
    const std::vector<int64_t>& dst_shape = ShapeOf(dst);
    if (src_shape.size() > dst_shape.size()) {
        Fail("cannot broadcast '" + src.Name() + "' into '" + dst.Name() + "'");
    }

    std::vector<std::string> indices;
    if (src_shape.empty()) {
        return indices;
    }

    const size_t rank_gap = dst_shape.size() - src_shape.size();
    for (size_t i = 0; i < src_shape.size(); ++i) {
        const int64_t src_dim = src_shape[i];
        const int64_t dst_dim = dst_shape[rank_gap + i];
        if (src_dim == dst_dim) {
            indices.push_back(dst_indices[rank_gap + i]);
        } else if (src_dim == 1) {
            indices.push_back(EmitIndexConst(0));
        } else {
            Fail("incompatible broadcast from '" + src.Name() + "' to '" + dst.Name() + "'");
        }
    }
    return indices;
}

void ModuleEmitter::EmitLoopNest(const std::vector<int64_t>& shape,
                                 size_t dim,
                                 std::vector<std::string>& indices,
                                 const std::function<void(const std::vector<std::string>&)>& body) {
    if (dim == shape.size()) {
        body(indices);
        return;
    }

    const std::string lb = EmitIndexConst(0);
    const std::string ub = EmitIndexConst(shape[dim]);
    const std::string step = EmitIndexConst(1);
    const std::string iv = NewSsa("i");

    EmitLine("scf.for " + iv + " = " + lb + " to " + ub + " step " + step + " {");
    ++indent_;
    indices.push_back(iv);
    EmitLoopNest(shape, dim + 1, indices, body);
    indices.pop_back();
    --indent_;
    EmitLine("}");
}

std::string ModuleEmitter::EmitAddLike(const std::string& lhs,
                                       const std::string& rhs,
                                       TensorElemType elem_type,
                                       std::string_view hint) {
    const std::string out = NewSsa(hint);
    if (IsFloatType(elem_type)) {
        EmitLine(out + " = arith.addf " + lhs + ", " + rhs + " : " + ElemTypeToMlir(elem_type));
        return out;
    }
    if (elem_type == TensorElemType::kInt32 || elem_type == TensorElemType::kInt64) {
        EmitLine(out + " = arith.addi " + lhs + ", " + rhs + " : " + ElemTypeToMlir(elem_type));
        return out;
    }
    Fail("unsupported add type");
}

std::string ModuleEmitter::EmitMulLike(const std::string& lhs,
                                       const std::string& rhs,
                                       TensorElemType elem_type,
                                       std::string_view hint) {
    const std::string out = NewSsa(hint);
    if (IsFloatType(elem_type)) {
        EmitLine(out + " = arith.mulf " + lhs + ", " + rhs + " : " + ElemTypeToMlir(elem_type));
        return out;
    }
    if (elem_type == TensorElemType::kInt32 || elem_type == TensorElemType::kInt64) {
        EmitLine(out + " = arith.muli " + lhs + ", " + rhs + " : " + ElemTypeToMlir(elem_type));
        return out;
    }
    Fail("unsupported mul type");
}

} // namespace tc::detail
