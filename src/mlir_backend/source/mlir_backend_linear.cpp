#include "mlir_backend_internal.hpp"

namespace tc::detail {

void ModuleEmitter::EmitMatMul(const Operation& op) {
    if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
        Fail(op.Name() + ": expected 2 inputs and 1 output");
    }

    const Value& a = *op.Inputs()[0];
    const Value& b = *op.Inputs()[1];
    const Value& y = *op.Outputs()[0];
    const TensorType& a_type = RequireTensorType(a);
    const TensorType& b_type = RequireTensorType(b);
    const TensorType& y_type = RequireTensorType(y);
    if (a_type.Shape().size() != 2 || b_type.Shape().size() != 2 || y_type.Shape().size() != 2) {
        Fail(op.Name() + ": MatMul currently supports rank-2 tensors only");
    }
    if (a_type.Shape()[1] != b_type.Shape()[0]) {
        Fail(op.Name() + ": incompatible MatMul inner dimensions");
    }
    if (!IsFloatType(y_type.ElemType()) && y_type.ElemType() != TensorElemType::kInt32 && y_type.ElemType() != TensorElemType::kInt64) {
        Fail(op.Name() + ": unsupported MatMul element type");
    }

    const int64_t m = y_type.Shape()[0];
    const int64_t n = y_type.Shape()[1];
    const int64_t k = a_type.Shape()[1];
    const std::string scalar_memref_type = "memref<" + ElemTypeToMlir(y_type.ElemType()) + ">";

    std::vector<std::string> outer_indices;
    EmitLoopNest({m, n}, 0, outer_indices, [&](const std::vector<std::string>& ij) {
        const std::string acc_buf = NewSsa("acc");
        EmitLine(acc_buf + " = memref.alloca() : " + scalar_memref_type);
        const std::string zero = EmitNumericConst(y_type.ElemType(), 0.0);
        EmitStoreRaw(zero, acc_buf, scalar_memref_type, {});

        std::vector<std::string> inner_indices;
        EmitLoopNest({k}, 0, inner_indices, [&](const std::vector<std::string>& kk) {
            const std::string lhs = EmitLoadValue(a, {ij[0], kk[0]}, "a");
            const std::string rhs = EmitLoadValue(b, {kk[0], ij[1]}, "b");
            const std::string prod = EmitMulLike(lhs, rhs, y_type.ElemType(), "prod");
            const std::string cur = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "cur");
            const std::string next = EmitAddLike(cur, prod, y_type.ElemType(), "sum");
            EmitStoreRaw(next, acc_buf, scalar_memref_type, {});
        });

        const std::string final_value = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "final");
        EmitStoreValue(final_value, y, ij);
    });
}

void ModuleEmitter::EmitTranspose(const Operation& op) {
    if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
        Fail(op.Name() + ": expected 1 input and 1 output");
    }

    const Value& input = *op.Inputs()[0];
    const Value& output = *op.Outputs()[0];
    const std::vector<int64_t> perm = GetIntsAttr(op.Attrs(), "perm", {});
    const size_t rank = ShapeOf(output).size();
    if (ShapeOf(input).size() != rank) {
        Fail(op.Name() + ": input/output rank mismatch for Transpose");
    }

    std::vector<int64_t> effective_perm = perm;
    if (effective_perm.empty()) {
        effective_perm.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            effective_perm[i] = static_cast<int64_t>(rank - 1 - i);
        }
    }
    if (effective_perm.size() != rank) {
        Fail(op.Name() + ": invalid permutation rank");
    }

    std::vector<size_t> inverse_perm(rank);
    for (size_t out_axis = 0; out_axis < rank; ++out_axis) {
        const int64_t src_axis = effective_perm[out_axis];
        if (src_axis < 0 || src_axis >= static_cast<int64_t>(rank)) {
            Fail(op.Name() + ": invalid permutation axis");
        }
        inverse_perm[static_cast<size_t>(src_axis)] = out_axis;
    }

    std::vector<std::string> indices;
    EmitLoopNest(ShapeOf(output), 0, indices, [&](const std::vector<std::string>& out_indices) {
        std::vector<std::string> in_indices(rank);
        for (size_t src_axis = 0; src_axis < rank; ++src_axis) {
            in_indices[src_axis] = out_indices[inverse_perm[src_axis]];
        }
        const std::string loaded = EmitLoadValue(input, in_indices, "transpose_in");
        EmitStoreValue(loaded, output, out_indices);
    });
}

void ModuleEmitter::EmitGemm(const Operation& op) {
    if ((op.Inputs().size() != 2 && op.Inputs().size() != 3) || op.Outputs().size() != 1) {
        Fail(op.Name() + ": expected 2 or 3 inputs and 1 output");
    }

    const Value& a = *op.Inputs()[0];
    const Value& b = *op.Inputs()[1];
    const Value* c = op.Inputs().size() == 3 ? op.Inputs()[2] : nullptr;
    const Value& y = *op.Outputs()[0];

    const TensorType& a_type = RequireTensorType(a);
    const TensorType& b_type = RequireTensorType(b);
    const TensorType& y_type = RequireTensorType(y);
    if (a_type.Shape().size() != 2 || b_type.Shape().size() != 2 || y_type.Shape().size() != 2) {
        Fail(op.Name() + ": Gemm currently supports rank-2 tensors only");
    }
    if (!IsFloatType(y_type.ElemType())) {
        Fail(op.Name() + ": Gemm currently supports floating-point tensors only");
    }

    const int64_t trans_a = GetIntAttr(op.Attrs(), "transA", 0);
    const int64_t trans_b = GetIntAttr(op.Attrs(), "transB", 0);
    const float alpha = GetFloatAttr(op.Attrs(), "alpha", 1.0f);
    const float beta = GetFloatAttr(op.Attrs(), "beta", 1.0f);

    const int64_t a_m = trans_a ? a_type.Shape()[1] : a_type.Shape()[0];
    const int64_t a_k = trans_a ? a_type.Shape()[0] : a_type.Shape()[1];
    const int64_t b_k = trans_b ? b_type.Shape()[1] : b_type.Shape()[0];
    const int64_t b_n = trans_b ? b_type.Shape()[0] : b_type.Shape()[1];
    if (a_k != b_k) {
        Fail(op.Name() + ": Gemm inner dimensions mismatch");
    }
    if (y_type.Shape()[0] != a_m || y_type.Shape()[1] != b_n) {
        Fail(op.Name() + ": Gemm output shape mismatch");
    }

    const std::string scalar_memref_type = "memref<" + ElemTypeToMlir(y_type.ElemType()) + ">";
    std::vector<std::string> outer_indices;
    EmitLoopNest({a_m, b_n}, 0, outer_indices, [&](const std::vector<std::string>& ij) {
        const std::string acc_buf = NewSsa("acc");
        EmitLine(acc_buf + " = memref.alloca() : " + scalar_memref_type);
        const std::string zero = EmitNumericConst(y_type.ElemType(), 0.0);
        EmitStoreRaw(zero, acc_buf, scalar_memref_type, {});

        std::vector<std::string> inner_indices;
        EmitLoopNest({a_k}, 0, inner_indices, [&](const std::vector<std::string>& kk) {
            const std::vector<std::string> a_idx = trans_a ? std::vector<std::string>{kk[0], ij[0]} : std::vector<std::string>{ij[0], kk[0]};
            const std::vector<std::string> b_idx = trans_b ? std::vector<std::string>{ij[1], kk[0]} : std::vector<std::string>{kk[0], ij[1]};
            const std::string lhs = EmitLoadValue(a, a_idx, "a");
            const std::string rhs = EmitLoadValue(b, b_idx, "b");
            const std::string prod = EmitMulLike(lhs, rhs, y_type.ElemType(), "prod");
            const std::string cur = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "cur");
            const std::string next = EmitAddLike(cur, prod, y_type.ElemType(), "sum");
            EmitStoreRaw(next, acc_buf, scalar_memref_type, {});
        });

        std::string result = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "gemm_acc");
        if (alpha != 1.0f) {
            const std::string alpha_cst = EmitNumericConst(y_type.ElemType(), alpha);
            result = EmitMulLike(result, alpha_cst, y_type.ElemType(), "alpha_scaled");
        }

        if (c != nullptr) {
            std::string c_value = EmitLoadValue(*c, BroadcastIndices(*c, y, ij), "c_bias");
            if (beta != 1.0f) {
                const std::string beta_cst = EmitNumericConst(y_type.ElemType(), beta);
                c_value = EmitMulLike(c_value, beta_cst, y_type.ElemType(), "beta_scaled");
            }
            result = EmitAddLike(result, c_value, y_type.ElemType(), "gemm_out");
        }

        EmitStoreValue(result, y, ij);
    });
}

} // namespace tc::detail
