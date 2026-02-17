#include "graph/attribute.hpp"

#include <stdexcept>

namespace tc {

Attribute::Attribute(const std::string& name, Value value)
    : name_{name}, value_{std::move(value)} {
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

const Attribute* FindAttribute(const AttributeMap& attrs, const std::string& name) {
    auto it = attrs.find(name);
    if (it == attrs.end()) return nullptr;
    return &it->second;
}

} // namespace tc