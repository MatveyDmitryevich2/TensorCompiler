#include "mlir_backend_internal.hpp"

namespace tc::detail {

void ModuleEmitter::EmitConv(const Operation& op) {
    if ((op.Inputs().size() != 2 && op.Inputs().size() != 3) || op.Outputs().size() != 1) {
        Fail(op.Name() + ": expected 2 or 3 inputs and 1 output");
    }

    const Value& x = *op.Inputs()[0];
    const Value& w = *op.Inputs()[1];
    const Value* bias = op.Inputs().size() == 3 ? op.Inputs()[2] : nullptr;
    const Value& y = *op.Outputs()[0];

    const TensorType& x_type = RequireTensorType(x);
    const TensorType& w_type = RequireTensorType(w);
    const TensorType& y_type = RequireTensorType(y);
    if (x_type.Shape().size() != 4 || w_type.Shape().size() != 4 || y_type.Shape().size() != 4) {
        Fail(op.Name() + ": Conv currently supports rank-4 tensors only");
    }
    if (!IsFloatType(y_type.ElemType())) {
        Fail(op.Name() + ": Conv currently supports floating-point tensors only");
    }

    std::vector<int64_t> pads = GetIntsAttr(op.Attrs(), "pads", {0, 0, 0, 0});
    if (pads.size() == 2) {
        pads = {pads[0], pads[1], pads[0], pads[1]};
    }
    if (pads.size() != 4) {
        Fail(op.Name() + ": pads attribute must have size 2 or 4");
    }
    const std::vector<int64_t> strides = GetIntsAttr(op.Attrs(), "strides", {1, 1});
    const std::vector<int64_t> dilations = GetIntsAttr(op.Attrs(), "dilations", {1, 1});
    if (strides.size() != 2 || dilations.size() != 2) {
        Fail(op.Name() + ": strides/dilations must have size 2");
    }
    const int64_t group = GetIntAttr(op.Attrs(), "group", 1);
    if (group <= 0) {
        Fail(op.Name() + ": group must be positive");
    }

    const int64_t n = x_type.Shape()[0];
    const int64_t c = x_type.Shape()[1];
    const int64_t h = x_type.Shape()[2];
    const int64_t width = x_type.Shape()[3];
    const int64_t out_channels = w_type.Shape()[0];
    const int64_t channels_per_group = w_type.Shape()[1];
    const int64_t kernel_h = w_type.Shape()[2];
    const int64_t kernel_w = w_type.Shape()[3];
    const int64_t out_h = y_type.Shape()[2];
    const int64_t out_w = y_type.Shape()[3];

    if (c != channels_per_group * group) {
        Fail(op.Name() + ": input channels do not match weights/group");
    }
    if (out_channels % group != 0) {
        Fail(op.Name() + ": output channels are not divisible by group");
    }
    if (bias != nullptr) {
        const TensorType& bias_type = RequireTensorType(*bias);
        if (bias_type.Shape().size() != 1 || bias_type.Shape()[0] != out_channels) {
            Fail(op.Name() + ": bias must have shape [out_channels]");
        }
    }

    const int64_t out_channels_per_group = out_channels / group;
    const std::string scalar_memref_type = "memref<" + ElemTypeToMlir(y_type.ElemType()) + ">";

    std::vector<std::string> outer_indices;
    EmitLoopNest({n, group, out_channels_per_group, out_h, out_w}, 0, outer_indices, [&](const std::vector<std::string>& ivs) {
        const std::string oc_group_mul = NewSsa("oc_group_mul");
        const std::string ocg_const = EmitIndexConst(out_channels_per_group);
        EmitLine(oc_group_mul + " = arith.muli " + ivs[1] + ", " + ocg_const + " : index");
        const std::string oc = NewSsa("oc");
        EmitLine(oc + " = arith.addi " + oc_group_mul + ", " + ivs[2] + " : index");

        const std::string c_group_mul = NewSsa("c_group_mul");
        const std::string cpg_const = EmitIndexConst(channels_per_group);
        EmitLine(c_group_mul + " = arith.muli " + ivs[1] + ", " + cpg_const + " : index");

        const std::string acc_buf = NewSsa("acc");
        EmitLine(acc_buf + " = memref.alloca() : " + scalar_memref_type);
        const std::string zero = EmitNumericConst(y_type.ElemType(), 0.0);
        EmitStoreRaw(zero, acc_buf, scalar_memref_type, {});

        std::vector<std::string> reduce_indices;
        EmitLoopNest({channels_per_group, kernel_h, kernel_w}, 0, reduce_indices, [&](const std::vector<std::string>& r) {
            const std::string in_c = NewSsa("in_c");
            EmitLine(in_c + " = arith.addi " + c_group_mul + ", " + r[0] + " : index");

            const std::string oh_mul = NewSsa("oh_mul");
            const std::string sh = EmitIndexConst(strides[0]);
            EmitLine(oh_mul + " = arith.muli " + ivs[3] + ", " + sh + " : index");
            const std::string kh_dil = NewSsa("kh_dil");
            const std::string dh = EmitIndexConst(dilations[0]);
            EmitLine(kh_dil + " = arith.muli " + r[1] + ", " + dh + " : index");
            const std::string ih_tmp = NewSsa("ih_tmp");
            const std::string pad_t = EmitIndexConst(pads[0]);
            EmitLine(ih_tmp + " = arith.subi " + oh_mul + ", " + pad_t + " : index");
            const std::string ih = NewSsa("ih");
            EmitLine(ih + " = arith.addi " + ih_tmp + ", " + kh_dil + " : index");

            const std::string ow_mul = NewSsa("ow_mul");
            const std::string sw = EmitIndexConst(strides[1]);
            EmitLine(ow_mul + " = arith.muli " + ivs[4] + ", " + sw + " : index");
            const std::string kw_dil = NewSsa("kw_dil");
            const std::string dw = EmitIndexConst(dilations[1]);
            EmitLine(kw_dil + " = arith.muli " + r[2] + ", " + dw + " : index");
            const std::string iw_tmp = NewSsa("iw_tmp");
            const std::string pad_l = EmitIndexConst(pads[1]);
            EmitLine(iw_tmp + " = arith.subi " + ow_mul + ", " + pad_l + " : index");
            const std::string iw = NewSsa("iw");
            EmitLine(iw + " = arith.addi " + iw_tmp + ", " + kw_dil + " : index");

            const std::string zero_idx = EmitIndexConst(0);
            const std::string h_idx = EmitIndexConst(h);
            const std::string w_idx = EmitIndexConst(width);
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
            const std::string prod = EmitMulLike(x_val, w_val, y_type.ElemType(), "prod");
            const std::string cur = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "cur");
            const std::string next = EmitAddLike(cur, prod, y_type.ElemType(), "sum");
            EmitStoreRaw(next, acc_buf, scalar_memref_type, {});
            --indent_;
            EmitLine("}");
        });

        std::string out_value = EmitLoadRaw(acc_buf, scalar_memref_type, {}, "conv_out");
        if (bias != nullptr) {
            const std::string b = EmitLoadValue(*bias, {oc}, "bias");
            out_value = EmitAddLike(out_value, b, y_type.ElemType(), "biased");
        }
        EmitStoreValue(out_value, y, {ivs[0], oc, ivs[3], ivs[4]});
    });
}

} // namespace tc::detail
