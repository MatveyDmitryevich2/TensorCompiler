#include "mlir_backend_internal.hpp"

namespace tc::detail {

void ModuleEmitter::EmitElementwiseBinary(const Operation& op, bool is_add) {
    RequireArity(op, 2, 1);

    const Value& lhs_value = *op.Inputs()[0];
    const Value& rhs_value = *op.Inputs()[1];
    const Value& out_value = *op.Outputs()[0];
    const TensorElemType elem_type = RequireTensorType(out_value).ElemType();

    std::vector<std::string> indices;
    indices.reserve(ShapeOf(out_value).size());
    EmitLoopNest(ShapeOf(out_value), 0, indices, [&](const std::vector<std::string>& ivs) {
        const std::string lhs = EmitLoadValue(lhs_value, BroadcastIndices(lhs_value, out_value, ivs), "lhs");
        const std::string rhs = EmitLoadValue(rhs_value, BroadcastIndices(rhs_value, out_value, ivs), "rhs");
        const std::string result = is_add
            ? EmitAddLike(lhs, rhs, elem_type, "add")
            : EmitMulLike(lhs, rhs, elem_type, "mul");
        EmitStoreValue(result, out_value, ivs);
    });
}

void ModuleEmitter::EmitRelu(const Operation& op) {
    RequireArity(op, 1, 1);

    const Value& input = *op.Inputs()[0];
    const Value& output = *op.Outputs()[0];
    const TensorElemType elem_type = RequireTensorType(output).ElemType();
    if (!IsFloatType(elem_type) && !IsIntegerLikeType(elem_type)) {
        Fail(op.Name() + ": unsupported Relu element type");
    }
    if (elem_type == TensorElemType::kBool) {
        Fail(op.Name() + ": bool Relu is not supported");
    }

    std::vector<std::string> indices;
    indices.reserve(ShapeOf(output).size());
    EmitLoopNest(ShapeOf(output), 0, indices, [&](const std::vector<std::string>& ivs) {
        const std::string arg = EmitLoadValue(input, BroadcastIndices(input, output, ivs), "relu_in");
        const std::string zero = EmitNumericConst(elem_type, 0.0);
        const std::string result = NewSsa("relu");
        if (IsFloatType(elem_type)) {
            EmitLine(result + " = arith.maximumf " + arg + ", " + zero + " : " + ElemTypeToMlir(elem_type));
        } else {
            EmitLine(result + " = arith.maxsi " + arg + ", " + zero + " : " + ElemTypeToMlir(elem_type));
        }
        EmitStoreValue(result, output, ivs);
    });
}

} // namespace tc::detail
