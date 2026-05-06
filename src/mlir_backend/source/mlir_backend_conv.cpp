#include "mlir_backend_internal.hpp"

namespace tc::detail {

namespace {

struct ConvAttrs {
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    int64_t group;
};

struct ConvShape {
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t out_channels;
    int64_t channels_per_group;
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t out_h;
    int64_t out_w;
    int64_t out_channels_per_group;
};

ConvAttrs ReadConvAttrs(const Operation& op) {
    std::vector<int64_t> pads = GetAttr<std::vector<int64_t>>(op.Attrs(), "pads", {0, 0, 0, 0});
    if (pads.size() == 2) {
        pads = {pads[0], pads[1], pads[0], pads[1]};
    }

    ConvAttrs attrs{
        std::move(pads),
        GetAttr<std::vector<int64_t>>(op.Attrs(), "strides", {1, 1}),
        GetAttr<std::vector<int64_t>>(op.Attrs(), "dilations", {1, 1}),
        GetAttr<int64_t>(op.Attrs(), "group", 1)
    };

    if (attrs.pads.size() != 4 || attrs.strides.size() != 2 ||
        attrs.dilations.size() != 2 || attrs.group <= 0) {
        Fail(op.Name() + ": invalid Conv attributes");
    }
    return attrs;
}

ConvShape ReadConvShape(const Operation& op,
                        const TensorType& x_type,
                        const TensorType& w_type,
                        const TensorType& y_type,
                        int64_t group) {
    ConvShape shape{
        x_type.Shape()[0],
        x_type.Shape()[1],
        x_type.Shape()[2],
        x_type.Shape()[3],
        w_type.Shape()[0],
        w_type.Shape()[1],
        w_type.Shape()[2],
        w_type.Shape()[3],
        y_type.Shape()[2],
        y_type.Shape()[3],
        0
    };

    if (shape.c != shape.channels_per_group * group) {
        Fail(op.Name() + ": input channels do not match weights/group");
    }
    if (shape.out_channels % group != 0) {
        Fail(op.Name() + ": output channels are not divisible by group");
    }
    shape.out_channels_per_group = shape.out_channels / group;
    return shape;
}

void ValidateBias(const Operation& op, const Value* bias, int64_t out_channels) {
    if (bias == nullptr) {
        return;
    }

    const TensorType& bias_type = RequireTensorType(*bias);
    if (bias_type.Shape().size() != 1 || bias_type.Shape()[0] != out_channels) {
        Fail(op.Name() + ": bias must have shape [out_channels]");
    }
}

} // namespace

void ModuleEmitter::EmitConv(const Operation& op) {
    RequireInputRange(op, 2, 3, 1);

    const Value& x = *op.Inputs()[0];
    const Value& w = *op.Inputs()[1];
    const Value* bias = op.Inputs().size() == 3 ? op.Inputs()[2] : nullptr;
    const Value& y = *op.Outputs()[0];

    const TensorType& x_type = RequireTensorType(x);
    const TensorType& w_type = RequireTensorType(w);
    const TensorType& y_type = RequireTensorType(y);
    RequireRank(op, x_type, 4, "input");
    RequireRank(op, w_type, 4, "weights");
    RequireRank(op, y_type, 4, "output");
    if (!IsFloatType(y_type.ElemType())) {
        Fail(op.Name() + ": Conv currently supports floating-point tensors only");
    }

    const ConvAttrs attrs = ReadConvAttrs(op);
    const ConvShape shape = ReadConvShape(op, x_type, w_type, y_type, attrs.group);
    ValidateBias(op, bias, shape.out_channels);

    const TensorElemType elem_type = y_type.ElemType();
    const std::string scalar_memref_type = ScalarMemRefType(elem_type);

    std::vector<std::string> outer_indices;
    EmitLoopNest({shape.n, attrs.group, shape.out_channels_per_group, shape.out_h, shape.out_w}, 0, outer_indices, [&](const std::vector<std::string>& ivs) {
        const std::string oc_base = EmitIndexMul(ivs[1], EmitIndexConst(shape.out_channels_per_group), "oc_base");
        const std::string oc = EmitIndexAdd(oc_base, ivs[2], "oc");
        const std::string c_base = EmitIndexMul(ivs[1], EmitIndexConst(shape.channels_per_group), "c_base");

        const std::string acc_buf = EmitScalarAlloca(elem_type, "acc");
        EmitZeroScalar(acc_buf, elem_type);

        std::vector<std::string> reduce_indices;
        EmitLoopNest({shape.channels_per_group, shape.kernel_h, shape.kernel_w}, 0, reduce_indices, [&](const std::vector<std::string>& r) {
            const std::string in_c = EmitIndexAdd(c_base, r[0], "in_c");

            const std::string oh_mul = EmitIndexMul(ivs[3], EmitIndexConst(attrs.strides[0]), "oh_mul");
            const std::string kh_dil = EmitIndexMul(r[1], EmitIndexConst(attrs.dilations[0]), "kh_dil");
            const std::string ih_tmp = EmitIndexSub(oh_mul, EmitIndexConst(attrs.pads[0]), "ih_tmp");
            const std::string ih = EmitIndexAdd(ih_tmp, kh_dil, "ih");

            const std::string ow_mul = EmitIndexMul(ivs[4], EmitIndexConst(attrs.strides[1]), "ow_mul");
            const std::string kw_dil = EmitIndexMul(r[2], EmitIndexConst(attrs.dilations[1]), "kw_dil");
            const std::string iw_tmp = EmitIndexSub(ow_mul, EmitIndexConst(attrs.pads[1]), "iw_tmp");
            const std::string iw = EmitIndexAdd(iw_tmp, kw_dil, "iw");

            const std::string zero_idx = EmitIndexConst(0);
            const std::string h_idx = EmitIndexConst(shape.h);
            const std::string w_idx = EmitIndexConst(shape.w);
            const std::string ih_ge_0 = NewSsa("ih_ge_0");
            EmitLine(ih_ge_0 + " = arith.cmpi sge, " + ih + ", " + zero_idx + " : index");
            const std::string ih_lt_h = NewSsa("ih_lt_h");
            EmitLine(ih_lt_h + " = arith.cmpi slt, " + ih + ", " + h_idx + " : index");
            const std::string iw_ge_0 = NewSsa("iw_ge_0");
            EmitLine(iw_ge_0 + " = arith.cmpi sge, " + iw + ", " + zero_idx + " : index");
            const std::string iw_lt_w = NewSsa("iw_lt_w");
            EmitLine(iw_lt_w + " = arith.cmpi slt, " + iw + ", " + w_idx + " : index");
            const std::string in_h = NewSsa("in_h");
            EmitLine(in_h + " = arith.andi " + ih_ge_0 + ", " + ih_lt_h + " : i1");
            const std::string in_w = NewSsa("in_w");
            EmitLine(in_w + " = arith.andi " + iw_ge_0 + ", " + iw_lt_w + " : i1");
            const std::string in_bounds = NewSsa("in_bounds");
            EmitLine(in_bounds + " = arith.andi " + in_h + ", " + in_w + " : i1");

            EmitLine("scf.if " + in_bounds + " {");
            ++indent_;
            const std::string x_val = EmitLoadValue(x, {ivs[0], in_c, ih, iw}, "x");
            const std::string w_val = EmitLoadValue(w, {oc, r[0], r[1], r[2]}, "w");
            const std::string prod = EmitMulLike(x_val, w_val, elem_type, "prod");
            const std::string cur = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "cur");
            const std::string next = EmitAddLike(cur, prod, elem_type, "sum");
            EmitStoreRaw(next, acc_buf, scalar_memref_type, {});
            --indent_;
            EmitLine("}");
        });

        std::string out_value = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "conv_out");
        if (bias != nullptr) {
            const std::string b = EmitLoadValue(*bias, {oc}, "bias");
            out_value = EmitAddLike(out_value, b, elem_type, "biased");
        }
        EmitStoreValue(out_value, y, {ivs[0], oc, ivs[3], ivs[4]});
    });
}

} // namespace tc::detail
