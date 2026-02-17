#include "graph/attribute.hpp"

#include <stdexcept>

namespace tc {

Attribute::Attribute(const std::string& name, Value value) : name_{name},
                                                             value_{std::move(value)} {
    if (name_.empty()) {
        throw std::runtime_error{"Attribute: empty name"};
    }
}

int64_t Attribute::AsInt() const {
    const auto* p = std::get_if<int64_t>(&value_);
    if (!p) throw std::runtime_error{"Attribute '" + name_ + "' is not INT"};
    return *p;
}

float Attribute::AsFloat() const {
    const auto* p = std::get_if<float>(&value_);
    if (!p) throw std::runtime_error{"Attribute '" + name_ + "' is not FLOAT"};
    return *p;
}

const std::string& Attribute::AsString() const {
    const auto* p = std::get_if<std::string>(&value_);
    if (!p) throw std::runtime_error{"Attribute '" + name_ + "' is not STRING"};
    return *p;
}

const std::vector<int64_t>& Attribute::AsInts() const {
    const auto* p = std::get_if<std::vector<int64_t>>(&value_);
    if (!p) throw std::runtime_error{"Attribute '" + name_ + "' is not INTS"};
    return *p;
}

const std::vector<float>& Attribute::AsFloats() const {
    const auto* p = std::get_if<std::vector<float>>(&value_);
    if (!p) throw std::runtime_error{"Attribute '" + name_ + "' is not FLOATS"};
    return *p;
}

const std::vector<std::string>& Attribute::AsStrings() const {
    const auto* p = std::get_if<std::vector<std::string>>(&value_);
    if (!p) throw std::runtime_error{"Attribute '" + name_ + "' is not STRINGS"};
    return *p;
}

AttributeMap ParseAttributes(const onnx::NodeProto& g_node) {
    AttributeMap out;

    for (const auto& a : g_node.attribute()) {
        const std::string& name = a.name();

        switch (a.type()) {
            case onnx::AttributeProto::INT:
                out.emplace(name, Attribute{name, static_cast<int64_t>(a.i())});
                break;
            case onnx::AttributeProto::FLOAT:
                out.emplace(name, Attribute{name, a.f()});
                break;
            case onnx::AttributeProto::STRING:
                out.emplace(name, Attribute{name, a.s()});
                break;
            case onnx::AttributeProto::INTS: {
                std::vector<int64_t> vec_num;
                for (auto x : a.ints()) vec_num.push_back(static_cast<int64_t>(x));
                out.emplace(name, Attribute{name, std::move(vec_num)});
                break;
            }
            case onnx::AttributeProto::FLOATS: {
                std::vector<float> vec_num;
                vec_num.reserve(static_cast<size_t>(a.floats_size()));
                for (auto x : a.floats()) vec_num.push_back(x);
                out.emplace(name, Attribute{name, std::move(vec_num)});
                break;
            }
            case onnx::AttributeProto::STRINGS: {
                std::vector<std::string> vec_num;
                vec_num.reserve(static_cast<size_t>(a.strings_size()));
                for (const auto& s : a.strings()) vec_num.push_back(s);
                out.emplace(name, Attribute{name, std::move(vec_num)});
                break;
            }

            default:
                throw std::runtime_error{"Unsupported ONNX attribute type: '" + name + "'"};
        }
    }

    return out;
}

const Attribute* FindAttribute(const AttributeMap& attrs, const std::string& name) {
    auto it = attrs.find(name);
    if (it == attrs.end()) return nullptr;
    return &it->second;
}

} // namespace tc
