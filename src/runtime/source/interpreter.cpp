#include "runtime/interpreter.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tc::runtime {

namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error{"runtime: " + message};
}

const TensorType& RequireType(const Value& value) {
    if (!value.HasTensorType()) {
        Fail("value '" + value.Name() + "' has no tensor type");
    }
    return *value.MaybeTensorType();
}

size_t NumElements(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, [](size_t total, int64_t dim) {
        if (dim < 0) {
            Fail("dynamic shapes are not supported");
        }
        return total * static_cast<size_t>(dim);
    });
}

std::vector<size_t> Strides(const std::vector<int64_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * static_cast<size_t>(shape[i - 1]);
    }
    return strides;
}

size_t Offset(const std::vector<size_t>& strides, const std::vector<int64_t>& indices) {
    size_t out = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        out += static_cast<size_t>(indices[i]) * strides[i];
    }
    return out;
}

size_t Offset(const std::vector<size_t>& strides, std::initializer_list<int64_t> indices) {
    size_t out = 0;
    size_t i = 0;
    for (int64_t index : indices) {
        out += static_cast<size_t>(index) * strides[i++];
    }
    return out;
}

template <typename Fn>
void ForEachIndex(const std::vector<int64_t>& shape, size_t dim, std::vector<int64_t>* indices, Fn&& body) {
    if (dim == shape.size()) {
        body(*indices);
        return;
    }
    for (int64_t i = 0; i < shape[dim]; ++i) {
        indices->push_back(i);
        ForEachIndex(shape, dim + 1, indices, body);
        indices->pop_back();
    }
}

template <typename T>
T GetAttr(const AttributeMap& attrs, const std::string& name, T default_value) {
    auto it = attrs.find(name);
    if (it == attrs.end()) {
        return default_value;
    }
    return it->second.As<T>();
}

void RequireArity(const Operation& op, size_t inputs, size_t outputs) {
    if (op.Inputs().size() != inputs || op.Outputs().size() != outputs) {
        Fail(op.Name() + ": expected " + std::to_string(inputs) + " inputs and " +
             std::to_string(outputs) + " outputs");
    }
}

void RequireInputRange(const Operation& op, size_t min_inputs, size_t max_inputs, size_t outputs) {
    if (op.Inputs().size() < min_inputs || op.Inputs().size() > max_inputs || op.Outputs().size() != outputs) {
        Fail(op.Name() + ": invalid input/output count");
    }
}

void RequireRank(const Operation& op, const Tensor& tensor, size_t rank, const std::string& role) {
    if (tensor.shape.size() != rank) {
        Fail(op.Name() + ": " + role + " must have rank " + std::to_string(rank));
    }
}

const Tensor& RequireTensor(const TensorMap& values, const Value& value) {
    auto it = values.find(value.Name());
    if (it == values.end()) {
        Fail("missing tensor '" + value.Name() + "'");
    }
    return it->second;
}

Tensor& RequireTensor(TensorMap* values, const Value& value) {
    auto it = values->find(value.Name());
    if (it == values->end()) {
        Fail("missing tensor '" + value.Name() + "'");
    }
    return it->second;
}

std::vector<int64_t> BroadcastIndex(const Tensor& src,
                                    const Tensor& dst,
                                    const std::vector<int64_t>& dst_indices) {
    if (src.shape.size() > dst.shape.size()) {
        Fail("cannot broadcast tensor");
    }
    if (src.shape.empty()) {
        return {};
    }

    const size_t rank_gap = dst.shape.size() - src.shape.size();
    std::vector<int64_t> indices;
    indices.reserve(src.shape.size());
    for (size_t i = 0; i < src.shape.size(); ++i) {
        const int64_t src_dim = src.shape[i];
        const int64_t dst_dim = dst.shape[rank_gap + i];
        if (src_dim == dst_dim) {
            indices.push_back(dst_indices[rank_gap + i]);
        } else if (src_dim == 1) {
            indices.push_back(0);
        } else {
            Fail("incompatible broadcast");
        }
    }
    return indices;
}

Tensor MakeEmptyLike(const Value& value) {
    const TensorType& type = RequireType(value);
    if (type.ElemType() != TensorElemType::kFloat32) {
        Fail("only float32 tensors are supported by the runtime");
    }
    return Tensor{type.Shape(), std::vector<float>(NumElements(type.Shape()), 0.0f)};
}

Tensor TensorFromInitializer(const Value& value) {
    const TensorData& init = *value.InitializerData();
    if (init.type.ElemType() != TensorElemType::kFloat32) {
        Fail("only float32 initializers are supported by the runtime");
    }

    const size_t count = NumElements(init.type.Shape());
    if (init.raw.size() != count * sizeof(float)) {
        Fail("initializer '" + value.Name() + "' raw byte size mismatch");
    }

    Tensor tensor{init.type.Shape(), std::vector<float>(count)};
    if (count != 0) {
        std::memcpy(tensor.data.data(), init.raw.data(), count * sizeof(float));
    }
    return tensor;
}

template <typename Fn>
void RunElementwise(const Operation& op, TensorMap* values, Fn apply) {
    RequireArity(op, 2, 1);

    const Tensor& lhs = RequireTensor(*values, *op.Inputs()[0]);
    const Tensor& rhs = RequireTensor(*values, *op.Inputs()[1]);
    Tensor& out = RequireTensor(values, *op.Outputs()[0]);
    const auto lhs_strides = Strides(lhs.shape);
    const auto rhs_strides = Strides(rhs.shape);
    const auto out_strides = Strides(out.shape);

    std::vector<int64_t> indices;
    ForEachIndex(out.shape, 0, &indices, [&](const std::vector<int64_t>& out_idx) {
        const auto lhs_idx = BroadcastIndex(lhs, out, out_idx);
        const auto rhs_idx = BroadcastIndex(rhs, out, out_idx);
        const float l = lhs.data[Offset(lhs_strides, lhs_idx)];
        const float r = rhs.data[Offset(rhs_strides, rhs_idx)];
        out.data[Offset(out_strides, out_idx)] = apply(l, r);
    });
}

void RunRelu(const Operation& op, TensorMap* values) {
    RequireArity(op, 1, 1);

    const Tensor& input = RequireTensor(*values, *op.Inputs()[0]);
    Tensor& output = RequireTensor(values, *op.Outputs()[0]);
    if (input.data.size() != output.data.size()) {
        Fail(op.Name() + ": Relu shape mismatch");
    }

    std::transform(input.data.begin(), input.data.end(), output.data.begin(), [](float value) {
        return std::max(0.0f, value);
    });
}

void RunMatMul(const Operation& op, TensorMap* values) {
    RequireArity(op, 2, 1);

    const Tensor& a = RequireTensor(*values, *op.Inputs()[0]);
    const Tensor& b = RequireTensor(*values, *op.Inputs()[1]);
    Tensor& y = RequireTensor(values, *op.Outputs()[0]);
    RequireRank(op, a, 2, "lhs");
    RequireRank(op, b, 2, "rhs");
    RequireRank(op, y, 2, "output");

    const int64_t m = a.shape[0];
    const int64_t k = a.shape[1];
    const int64_t n = b.shape[1];
    if (b.shape[0] != k || y.shape[0] != m || y.shape[1] != n) {
        Fail(op.Name() + ": MatMul shape mismatch");
    }

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                acc += a.data[static_cast<size_t>(i * k + kk)] * b.data[static_cast<size_t>(kk * n + j)];
            }
            y.data[static_cast<size_t>(i * n + j)] = acc;
        }
    }
}

std::vector<int64_t> EffectivePerm(const Operation& op, size_t rank) {
    std::vector<int64_t> perm = GetAttr<std::vector<int64_t>>(op.Attrs(), "perm", {});
    if (perm.empty()) {
        perm.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            perm[i] = static_cast<int64_t>(rank - 1 - i);
        }
    }
    if (perm.size() != rank) {
        Fail(op.Name() + ": Transpose rank mismatch");
    }
    for (int64_t axis : perm) {
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            Fail(op.Name() + ": invalid transpose axis");
        }
    }
    return perm;
}

void RunTranspose(const Operation& op, TensorMap* values) {
    RequireArity(op, 1, 1);

    const Tensor& input = RequireTensor(*values, *op.Inputs()[0]);
    Tensor& output = RequireTensor(values, *op.Outputs()[0]);
    if (input.shape.size() != output.shape.size()) {
        Fail(op.Name() + ": Transpose rank mismatch");
    }

    const size_t rank = output.shape.size();
    const std::vector<int64_t> perm = EffectivePerm(op, rank);
    const auto input_strides = Strides(input.shape);
    const auto output_strides = Strides(output.shape);

    std::vector<int64_t> output_idx;
    ForEachIndex(output.shape, 0, &output_idx, [&](const std::vector<int64_t>& out_idx) {
        std::vector<int64_t> in_idx(rank);
        for (size_t out_axis = 0; out_axis < rank; ++out_axis) {
            in_idx[static_cast<size_t>(perm[out_axis])] = out_idx[out_axis];
        }
        output.data[Offset(output_strides, out_idx)] = input.data[Offset(input_strides, in_idx)];
    });
}

struct GemmAttrs {
    bool trans_a;
    bool trans_b;
    float alpha;
    float beta;
};

GemmAttrs ReadGemmAttrs(const Operation& op) {
    return GemmAttrs{
        GetAttr<int64_t>(op.Attrs(), "transA", 0) != 0,
        GetAttr<int64_t>(op.Attrs(), "transB", 0) != 0,
        GetAttr<float>(op.Attrs(), "alpha", 1.0f),
        GetAttr<float>(op.Attrs(), "beta", 1.0f)
    };
}

float MatrixValue(const Tensor& tensor, bool transposed, int64_t row, int64_t col) {
    return transposed
        ? tensor.data[static_cast<size_t>(col * tensor.shape[1] + row)]
        : tensor.data[static_cast<size_t>(row * tensor.shape[1] + col)];
}

void RunGemm(const Operation& op, TensorMap* values) {
    RequireInputRange(op, 2, 3, 1);

    const Tensor& a = RequireTensor(*values, *op.Inputs()[0]);
    const Tensor& b = RequireTensor(*values, *op.Inputs()[1]);
    const Tensor* c = op.Inputs().size() == 3 ? &RequireTensor(*values, *op.Inputs()[2]) : nullptr;
    Tensor& y = RequireTensor(values, *op.Outputs()[0]);
    RequireRank(op, a, 2, "A");
    RequireRank(op, b, 2, "B");
    RequireRank(op, y, 2, "output");

    const GemmAttrs attrs = ReadGemmAttrs(op);
    const int64_t m = attrs.trans_a ? a.shape[1] : a.shape[0];
    const int64_t k = attrs.trans_a ? a.shape[0] : a.shape[1];
    const int64_t n = attrs.trans_b ? b.shape[0] : b.shape[1];
    const int64_t b_k = attrs.trans_b ? b.shape[1] : b.shape[0];
    if (b_k != k || y.shape[0] != m || y.shape[1] != n) {
        Fail(op.Name() + ": Gemm shape mismatch");
    }

    const auto c_strides = c == nullptr ? std::vector<size_t>{} : Strides(c->shape);
    const auto y_strides = Strides(y.shape);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                acc += MatrixValue(a, attrs.trans_a, i, kk) * MatrixValue(b, attrs.trans_b, kk, j);
            }

            float result = attrs.alpha * acc;
            if (c != nullptr) {
                const std::vector<int64_t> y_idx{i, j};
                const auto c_idx = BroadcastIndex(*c, y, y_idx);
                result += attrs.beta * c->data[Offset(c_strides, c_idx)];
            }
            y.data[Offset(y_strides, {i, j})] = result;
        }
    }
}

struct ConvAttrs {
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    int64_t group;
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

struct ConvShape {
    int64_t batches;
    int64_t channels;
    int64_t height;
    int64_t width;
    int64_t out_channels;
    int64_t channels_per_group;
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t out_h;
    int64_t out_w;
    int64_t out_channels_per_group;
};

ConvShape ReadConvShape(const Operation& op, const Tensor& x, const Tensor& w, const Tensor& y, int64_t group) {
    ConvShape shape{
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        w.shape[0],
        w.shape[1],
        w.shape[2],
        w.shape[3],
        y.shape[2],
        y.shape[3],
        0
    };

    if (shape.channels != shape.channels_per_group * group || shape.out_channels % group != 0) {
        Fail(op.Name() + ": Conv channel/group mismatch");
    }
    shape.out_channels_per_group = shape.out_channels / group;
    return shape;
}

bool InsideImage(int64_t h, int64_t w, const ConvShape& shape) {
    return h >= 0 && h < shape.height && w >= 0 && w < shape.width;
}

float ConvWindowSum(const Tensor& x,
                    const Tensor& w,
                    const ConvAttrs& attrs,
                    const ConvShape& shape,
                    const std::vector<size_t>& x_strides,
                    const std::vector<size_t>& w_strides,
                    int64_t batch,
                    int64_t group,
                    int64_t oc,
                    int64_t oh,
                    int64_t ow) {
    float acc = 0.0f;
    for (int64_t cg = 0; cg < shape.channels_per_group; ++cg) {
        const int64_t ic = group * shape.channels_per_group + cg;
        for (int64_t kh = 0; kh < shape.kernel_h; ++kh) {
            const int64_t ih = oh * attrs.strides[0] - attrs.pads[0] + kh * attrs.dilations[0];
            for (int64_t kw = 0; kw < shape.kernel_w; ++kw) {
                const int64_t iw = ow * attrs.strides[1] - attrs.pads[1] + kw * attrs.dilations[1];
                if (!InsideImage(ih, iw, shape)) {
                    continue;
                }
                const float xv = x.data[Offset(x_strides, {batch, ic, ih, iw})];
                const float wv = w.data[Offset(w_strides, {oc, cg, kh, kw})];
                acc += xv * wv;
            }
        }
    }
    return acc;
}

void RunConv(const Operation& op, TensorMap* values) {
    RequireInputRange(op, 2, 3, 1);

    const Tensor& x = RequireTensor(*values, *op.Inputs()[0]);
    const Tensor& w = RequireTensor(*values, *op.Inputs()[1]);
    const Tensor* bias = op.Inputs().size() == 3 ? &RequireTensor(*values, *op.Inputs()[2]) : nullptr;
    Tensor& y = RequireTensor(values, *op.Outputs()[0]);
    RequireRank(op, x, 4, "input");
    RequireRank(op, w, 4, "weights");
    RequireRank(op, y, 4, "output");

    const ConvAttrs attrs = ReadConvAttrs(op);
    const ConvShape shape = ReadConvShape(op, x, w, y, attrs.group);
    const auto x_strides = Strides(x.shape);
    const auto w_strides = Strides(w.shape);
    const auto y_strides = Strides(y.shape);

    for (int64_t batch = 0; batch < shape.batches; ++batch) {
        for (int64_t group = 0; group < attrs.group; ++group) {
            for (int64_t ocg = 0; ocg < shape.out_channels_per_group; ++ocg) {
                const int64_t oc = group * shape.out_channels_per_group + ocg;
                for (int64_t oh = 0; oh < shape.out_h; ++oh) {
                    for (int64_t ow = 0; ow < shape.out_w; ++ow) {
                        float value = ConvWindowSum(x, w, attrs, shape, x_strides, w_strides, batch, group, oc, oh, ow);
                        if (bias != nullptr) {
                            value += bias->data[static_cast<size_t>(oc)];
                        }
                        y.data[Offset(y_strides, {batch, oc, oh, ow})] = value;
                    }
                }
            }
        }
    }
}

void RunOperation(const Operation& op, TensorMap* values) {
    switch (op.Type()) {
        case Operation::OpType::kAdd:
            RunElementwise(op, values, [](float lhs, float rhs) { return lhs + rhs; });
            return;
        case Operation::OpType::kMul:
            RunElementwise(op, values, [](float lhs, float rhs) { return lhs * rhs; });
            return;
        case Operation::OpType::kRelu:
            RunRelu(op, values);
            return;
        case Operation::OpType::kMatMul:
            RunMatMul(op, values);
            return;
        case Operation::OpType::kTranspose:
            RunTranspose(op, values);
            return;
        case Operation::OpType::kGemm:
            RunGemm(op, values);
            return;
        case Operation::OpType::kConv:
            RunConv(op, values);
            return;
    }
    Fail("unsupported operation");
}

void InitializeValue(const Value& value, TensorMap* values, std::vector<const Value*>* inputs, std::vector<const Value*>* outputs) {
    if (value.GetBelongsTo() == Value::BelongTo::kInitializer) {
        values->emplace(value.Name(), TensorFromInitializer(value));
    } else {
        values->emplace(value.Name(), MakeEmptyLike(value));
    }

    if (value.GetBelongsTo() == Value::BelongTo::kInput) {
        inputs->push_back(&value);
    }
    if (value.GetBelongsTo() == Value::BelongTo::kOutput) {
        outputs->push_back(&value);
    }
}

void ApplyInputs(const TensorMap& inputs, const std::vector<const Value*>& graph_inputs, TensorMap* values) {
    for (const Value* value : graph_inputs) {
        if (!inputs.contains(value->Name())) {
            Fail("missing graph input '" + value->Name() + "'");
        }
    }

    for (const auto& [name, tensor] : inputs) {
        auto it = values->find(name);
        if (it == values->end()) {
            Fail("input '" + name + "' does not exist in graph");
        }
        if (it->second.shape != tensor.shape) {
            Fail("input '" + name + "' shape mismatch: expected " +
                 ShapeToStr(it->second.shape) + ", got " + ShapeToStr(tensor.shape));
        }
        if (it->second.data.size() != tensor.data.size()) {
            Fail("input '" + name + "' data size mismatch");
        }
        it->second.data = tensor.data;
    }
}

} // namespace

TensorMap Interpreter::Run(const Graph& graph, const TensorMap& inputs) const {
    TensorMap values;
    std::vector<const Value*> graph_inputs;
    std::vector<const Value*> graph_outputs;

    for (const INode* node : graph) {
        if (const auto* value = dynamic_cast<const Value*>(node)) {
            InitializeValue(*value, &values, &graph_inputs, &graph_outputs);
        }
    }

    ApplyInputs(inputs, graph_inputs, &values);

    for (const INode* node : graph) {
        if (const auto* op = dynamic_cast<const Operation*>(node)) {
            RunOperation(*op, &values);
        }
    }

    TensorMap result;
    for (const Value* output : graph_outputs) {
        result.emplace(output->Name(), RequireTensor(values, *output));
    }
    return result;
}

} // namespace tc::runtime
