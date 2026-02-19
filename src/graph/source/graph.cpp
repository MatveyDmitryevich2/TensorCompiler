#include "graph/graph.hpp"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace tc {

namespace {

std::string EscapeDot(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': break;
            case '\t': out += "  "; break;
            default:   out += c; break;
        }
    }
    return out;
}

std::string OpTypeToStr(OpType op) {
    switch (op) {
        case OpType::kAdd:       return "Add";
        case OpType::kMul:       return "Mul";
        case OpType::kConv:      return "Conv";
        case OpType::kRelu:      return "Relu";
        case OpType::kMatMul:    return "MatMul";
        case OpType::kGemm:      return "Gemm";
        case OpType::kTranspose: return "Transpose";
    }
    return "<unknown>";
}

template <class T>
std::string JoinVec(const std::vector<T>& v, size_t max_items) {
    std::ostringstream oss;
    oss << "[";
    size_t n = v.size();
    size_t lim = (max_items == 0) ? n : std::min(n, max_items);
    for (size_t i = 0; i < lim; i++) {
        if (i) oss << ",";
        oss << v[i];
    }
    if (lim < n) oss << ",...";
    oss << "]";
    return oss.str();
}

std::string JoinVecStr(const std::vector<std::string>& v, size_t max_items) {
    std::ostringstream oss;
    oss << "[";
    size_t n = v.size();
    size_t lim = (max_items == 0) ? n : std::min(n, max_items);
    for (size_t i = 0; i < lim; i++) {
        if (i) oss << ",";
        oss << '"' << EscapeDot(v[i]) << '"';
    }
    if (lim < n) oss << ",...";
    oss << "]";
    return oss.str();
}

std::string AttrValueToStr(const Attribute::AttrValue& v, const DotOptions& opt) {
    return std::visit(
        [&](auto&& x) -> std::string {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, int64_t>) {
                return std::to_string(x);
            } else if constexpr (std::is_same_v<T, float>) {
                std::ostringstream oss;
                oss << x;
                return oss.str();
            } else if constexpr (std::is_same_v<T, std::string>) {
                return '"' + EscapeDot(x) + '"';
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                return JoinVec(x, opt.max_attr_items);
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                std::ostringstream oss;
                oss << "[";
                size_t n = x.size();
                size_t lim = (opt.max_attr_items == 0) ? n : std::min(n, opt.max_attr_items);
                for (size_t i = 0; i < lim; i++) {
                    if (i) oss << ",";
                    oss << x[i];
                }
                if (lim < n) oss << ",...";
                oss << "]";
                return oss.str();
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                return JoinVecStr(x, opt.max_attr_items);
            } else {
                return "<unknown>";
            }
        },
        v
    );
}

std::string AttrsToLabel(const AttributeMap& attrs, const DotOptions& opt) {
    if (!opt.show_attrs || attrs.empty()) return {};

    std::string out;
    out.reserve(256);

    size_t used = 0;
    size_t count = 0;

    for (const auto& kv : attrs) {
        if (opt.max_attr_items != 0 && count >= opt.max_attr_items) {
            out += "...\\l";
            break;
        }

        std::string line = kv.first + "=" + AttrValueToStr(kv.second.GetValue(), opt);
        out += EscapeDot(line);
        out += "\\l"; // left-justified new line

        used += line.size();
        count++;

        if (opt.max_attr_chars != 0 && used >= opt.max_attr_chars) {
            out += "...\\l";
            break;
        }
    }

    return out;
}

std::string ValueFillColor(Value::BelongTo b) {
    switch (b) {
        case Value::BelongTo::kInput:       return "#4FC3F7";
        case Value::BelongTo::kOutput:      return "#66BB6A";
        case Value::BelongTo::kInitializer: return "#FFCA28";
        case Value::BelongTo::kInternal:    return "#FFFFFF";
    }
    return "#FFFFFF";
}

} // namespace

std::string Graph::ToDot(const DotOptions& opt) const {
    std::unordered_map<const INode*, std::string> ids;
    ids.reserve(1024);

    auto id_of = [&](const INode* n) -> std::string {
        auto it = ids.find(n);
        if (it != ids.end()) return it->second;
        std::string id = "n" + std::to_string(reinterpret_cast<uintptr_t>(n));
        ids.emplace(n, id);
        return id;
    };

    std::ostringstream dot;
    dot << "digraph tc_graph {\n";
    if (opt.rank_left_to_right) dot << "  rankdir=LR;\n";
    dot << "  graph [fontname=\"Helvetica\"];\n";
    dot << "  node  [fontname=\"Helvetica\"];\n";
    dot << "  edge  [fontname=\"Helvetica\"];\n";

    for (const INode* n : *this) {
        if (const auto* v = dynamic_cast<const Value*>(n)) {
            if (!opt.show_values) continue;
            if (v->Name() == "<no name>") continue;

            dot << "  " << id_of(v)
                << " [shape=ellipse, style=filled, fillcolor=\"" << ValueFillColor(v->GetBelongsTo())
                << "\", label=\"" << EscapeDot(v->Name()) << "\"];\n";
            continue;
        }

        if (const auto* op = dynamic_cast<const Operation*>(n)) {
            std::string label = OpTypeToStr(op->Type());
            label += "\\n";
            label += op->Name();

            std::string attrs = AttrsToLabel(op->Attrs(), opt);
            if (!attrs.empty()) {
                label += "\\n";
                label += attrs;
            }

            dot << "  " << id_of(op)
                << " [shape=box, style=\"rounded,filled\", fillcolor=\"#B39DDB\""
                << ", labeljust=\"l\""
                << ", label=\"" << EscapeDot(label) << "\"];\n";
            continue;
        }
    }

    for (const INode* n : *this) {
        const auto* op = dynamic_cast<const Operation*>(n);
        if (!op) continue;

        for (size_t i = 0; i < op->Inputs().size(); i++) {
            const Value* v = op->Inputs()[i];
            if (!v) continue;
            if (!opt.show_values) continue;
            if (v->Name() == "<no name>") continue;

            dot << "  " << id_of(v) << " -> " << id_of(op);
            if (opt.show_edge_indices) dot << " [label=\"in" << i << "\"]";
            dot << ";\n";
        }

        for (size_t i = 0; i < op->Outputs().size(); i++) {
            const Value* v = op->Outputs()[i];
            if (!v) continue;
            if (!opt.show_values) continue;
            if (v->Name() == "<no name>") continue;

            dot << "  " << id_of(op) << " -> " << id_of(v);
            if (opt.show_edge_indices) dot << " [label=\"out" << i << "\"]";
            dot << ";\n";
        }
    }

    dot << "}\n";
    return dot.str();
}

} // namespace tc
