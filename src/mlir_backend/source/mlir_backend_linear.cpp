#include "mlir_backend_internal.hpp"

namespace tc::detail {

namespace {

struct GemmAttrs {
    bool trans_a;
    bool trans_b;
    float alpha;
    float beta;
};

GemmAttrs ReadGemmAttrs(const Operation& op) {
    return GemmAttrs{
        GetAttrOr<int64_t>(op.Attrs(), "transA", 0) != 0,
        GetAttrOr<int64_t>(op.Attrs(), "transB", 0) != 0,
        GetAttrOr<float>(op.Attrs(), "alpha", 1.0f),
        GetAttrOr<float>(op.Attrs(), "beta", 1.0f)
    };
}

std::vector<int64_t> EffectivePerm(const Operation& op, size_t rank) {
    std::vector<int64_t> perm = GetAttrOr<std::vector<int64_t>>(op.Attrs(), "perm", {});
    if (perm.empty()) {
        perm.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            perm[i] = static_cast<int64_t>(rank - 1 - i);
        }
    }
    if (perm.size() != rank) {
        Fail(op.Name() + ": invalid permutation rank");
    }
    return perm;
}

std::vector<size_t> InversePerm(const Operation& op, const std::vector<int64_t>& perm) {
    std::vector<size_t> inverse(perm.size());
    for (size_t out_axis = 0; out_axis < perm.size(); ++out_axis) {
        const int64_t src_axis = perm[out_axis];
        if (src_axis < 0 || src_axis >= static_cast<int64_t>(perm.size())) {
            Fail(op.Name() + ": invalid permutation axis");
        }
        inverse[static_cast<size_t>(src_axis)] = out_axis;
    }
    return inverse;
}

} // namespace

void ModuleEmitter::EmitMatMul(const Operation& op) {
    RequireArity(op, 2, 1);

    const Value& a = *op.Inputs()[0];
    const Value& b = *op.Inputs()[1];
    const Value& y = *op.Outputs()[0];
    const TensorType& a_type = RequireTensorType(a);
    const TensorType& b_type = RequireTensorType(b);
    const TensorType& y_type = RequireTensorType(y);
    RequireRank(op, a_type, 2, "lhs");
    RequireRank(op, b_type, 2, "rhs");
    RequireRank(op, y_type, 2, "output");
    if (a_type.Shape()[1] != b_type.Shape()[0]) {
        Fail(op.Name() + ": incompatible MatMul inner dimensions");
    }
    if (!IsFloatType(y_type.ElemType()) && y_type.ElemType() != TensorElemType::kInt32 && y_type.ElemType() != TensorElemType::kInt64) {
        Fail(op.Name() + ": unsupported MatMul element type");
    }

    const int64_t m = y_type.Shape()[0];
    const int64_t n = y_type.Shape()[1];
    const int64_t k = a_type.Shape()[1];
    const TensorElemType elem_type = y_type.ElemType();
    const std::string scalar_memref_type = ScalarMemRefType(elem_type);

    std::vector<std::string> outer_indices;
    outer_indices.reserve(2);
    EmitLoopNest({m, n}, 0, outer_indices, [&](const std::vector<std::string>& ij) {
        const std::string acc_buf = EmitScalarAlloca(elem_type, "acc");
        EmitZeroScalar(acc_buf, elem_type);

        std::vector<std::string> inner_indices;
        inner_indices.reserve(1);
        EmitLoopNest({k}, 0, inner_indices, [&](const std::vector<std::string>& kk) {
            const std::string lhs = EmitLoadValue(a, {ij[0], kk[0]}, "a");
            const std::string rhs = EmitLoadValue(b, {kk[0], ij[1]}, "b");
            const std::string prod = EmitMulLike(lhs, rhs, elem_type, "prod");
            const std::string cur = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "cur");
            const std::string next = EmitAddLike(cur, prod, elem_type, "sum");
            EmitStoreRaw(next, acc_buf, scalar_memref_type, {});
        });

        const std::string final_value = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "final");
        EmitStoreValue(final_value, y, ij);
    });
}

void ModuleEmitter::EmitTranspose(const Operation& op) {
    RequireArity(op, 1, 1);

    const Value& input = *op.Inputs()[0];
    const Value& output = *op.Outputs()[0];
    const size_t rank = ShapeOf(output).size();
    if (ShapeOf(input).size() != rank) {
        Fail(op.Name() + ": input/output rank mismatch for Transpose");
    }

    const std::vector<size_t> inverse_perm = InversePerm(op, EffectivePerm(op, rank));

    std::vector<std::string> indices;
    indices.reserve(rank);
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
    RequireInputRange(op, 2, 3, 1);

    const Value& a = *op.Inputs()[0];
    const Value& b = *op.Inputs()[1];
    const Value* c = op.Inputs().size() == 3 ? op.Inputs()[2] : nullptr;
    const Value& y = *op.Outputs()[0];

    const TensorType& a_type = RequireTensorType(a);
    const TensorType& b_type = RequireTensorType(b);
    const TensorType& y_type = RequireTensorType(y);
    RequireRank(op, a_type, 2, "A");
    RequireRank(op, b_type, 2, "B");
    RequireRank(op, y_type, 2, "output");
    if (!IsFloatType(y_type.ElemType())) {
        Fail(op.Name() + ": Gemm currently supports floating-point tensors only");
    }

    const GemmAttrs attrs = ReadGemmAttrs(op);

    const int64_t a_m = attrs.trans_a ? a_type.Shape()[1] : a_type.Shape()[0];
    const int64_t a_k = attrs.trans_a ? a_type.Shape()[0] : a_type.Shape()[1];
    const int64_t b_k = attrs.trans_b ? b_type.Shape()[1] : b_type.Shape()[0];
    const int64_t b_n = attrs.trans_b ? b_type.Shape()[0] : b_type.Shape()[1];
    if (a_k != b_k) {
        Fail(op.Name() + ": Gemm inner dimensions mismatch");
    }
    if (y_type.Shape()[0] != a_m || y_type.Shape()[1] != b_n) {
        Fail(op.Name() + ": Gemm output shape mismatch");
    }

    const TensorElemType elem_type = y_type.ElemType();
    const std::string scalar_memref_type = ScalarMemRefType(elem_type);
    std::vector<std::string> outer_indices;
    outer_indices.reserve(2);
    EmitLoopNest({a_m, b_n}, 0, outer_indices, [&](const std::vector<std::string>& ij) {
        const std::string acc_buf = EmitScalarAlloca(elem_type, "acc");
        EmitZeroScalar(acc_buf, elem_type);

        std::vector<std::string> inner_indices;
        inner_indices.reserve(1);
        EmitLoopNest({a_k}, 0, inner_indices, [&](const std::vector<std::string>& kk) {
            const std::vector<std::string> a_idx = attrs.trans_a ? std::vector<std::string>{kk[0], ij[0]} : std::vector<std::string>{ij[0], kk[0]};
            const std::vector<std::string> b_idx = attrs.trans_b ? std::vector<std::string>{ij[1], kk[0]} : std::vector<std::string>{kk[0], ij[1]};
            const std::string lhs = EmitLoadValue(a, a_idx, "a");
            const std::string rhs = EmitLoadValue(b, b_idx, "b");
            const std::string prod = EmitMulLike(lhs, rhs, elem_type, "prod");
            const std::string cur = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "cur");
            const std::string next = EmitAddLike(cur, prod, elem_type, "sum");
            EmitStoreRaw(next, acc_buf, scalar_memref_type, {});
        });

        std::string result = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "gemm_acc");
        if (attrs.alpha != 1.0f) {
            const std::string alpha_cst = EmitNumericConst(elem_type, attrs.alpha);
            result = EmitMulLike(result, alpha_cst, elem_type, "alpha_scaled");
        }

        if (c != nullptr) {
            std::string c_value = EmitLoadValue(*c, BroadcastIndices(*c, y, ij), "c_bias");
            if (attrs.beta != 1.0f) {
                const std::string beta_cst = EmitNumericConst(elem_type, attrs.beta);
                c_value = EmitMulLike(c_value, beta_cst, elem_type, "beta_scaled");
            }
            result = EmitAddLike(result, c_value, elem_type, "gemm_out");
        }

        EmitStoreValue(result, y, ij);
    });
}

} // namespace tc::detail
