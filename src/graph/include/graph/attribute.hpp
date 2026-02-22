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
    Attribute(const std::string& name, AttrValue value)
        : name_{name}, value_{std::move(value)} 
    {
        if (name_.empty()) {
            throw std::runtime_error{"Attribute: empty name"};
        }
    }

    const std::string& Name() const { return name_; }
    const AttrValue& GetValue() const { return value_; }
  private:
    template <typename T>
    constexpr std::string_view AttrTypeToStr() {
#define DEF_ATTR_TYPE_TO_STR(type_) else if (std::is_same_v<T, type_>) { return (#type_); }

        if (0) { return ""; }
        DEF_ATTR_TYPE_TO_STR(int64_t)
        DEF_ATTR_TYPE_TO_STR(float)
        DEF_ATTR_TYPE_TO_STR(std::string)
        DEF_ATTR_TYPE_TO_STR(std::vector<int64_t>)
        DEF_ATTR_TYPE_TO_STR(std::vector<float>)
        DEF_ATTR_TYPE_TO_STR(std::vector<std::string>)
        else { return "<unknown>"; }

#undef DEF_ATTR_TYPE_TO_STR
    }
  public:
    template <typename T>
    const T& As() const {
        const T* p = std::get_if<T>(&value_);
        if (p == nullptr) { 
            throw std::runtime_error{
                "Attribute '" + name_ + "' is not " + AttrTypeToStr<T>() 
            }; 
        }
        return *p;
    }
};

using AttributeMap = std::unordered_map<std::string, Attribute>;

} // namespace tc

#endif // ATTRIBUTE_HPP_
