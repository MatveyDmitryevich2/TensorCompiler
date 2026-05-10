#ifndef ATTRIBUTE_HPP_
#define ATTRIBUTE_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <string_view>

namespace tc {

class Attribute {
  public:
    using AttrValue = std::variant<
        int64_t,
        float,
        std::string,
        std::vector<int64_t>,
        std::vector<float>,
        std::vector<std::string>
    >;

  private:
    std::string name_;
    AttrValue value_;

  public:
    Attribute(std::string name, AttrValue value)
        : name_{std::move(name)}, value_{std::move(value)} {
        if (name_.empty()) {
            throw std::runtime_error{"Attribute: empty name"};
        }
    }

    const std::string& Name() const { return name_; }
    const AttrValue& GetValue() const { return value_; }
  private:
    template <typename T>
    static constexpr std::string_view AttrTypeToStr() {
        if constexpr (std::is_same_v<T, int64_t>) {
            return "int64_t";
        } else if constexpr (std::is_same_v<T, float>) {
            return "float";
        } else if constexpr (std::is_same_v<T, std::string>) {
            return "std::string";
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            return "std::vector<int64_t>";
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            return "std::vector<float>";
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            return "std::vector<std::string>";
        } else {
            return "<unknown>";
        }
    }
  public:
    template <typename T>
    const T& As() const {
        const T* p = std::get_if<T>(&value_);
        if (p == nullptr) {
            throw std::runtime_error(
                std::string("Attribute '") + name_ +
                "' is not " + std::string(AttrTypeToStr<T>())
            );
        }
        return *p;
    }
};

using AttributeMap = std::unordered_map<std::string, Attribute>;

template <typename T>
T GetAttrOr(const AttributeMap& attrs, const std::string& name, T default_value) {
    auto it = attrs.find(name);
    if (it == attrs.end()) {
        return default_value;
    }
    return it->second.As<T>();
}

} // namespace tc

#endif // ATTRIBUTE_HPP_
