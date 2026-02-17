#ifndef ATTRIBUTE_HPP_
#define ATTRIBUTE_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <cstddef>
#include <utility>

#include "onnx/onnx_pb.h"

namespace tc {

class Attribute {
  public:
    using Value = std::variant<
        int64_t,
        float,
        std::string,
        std::vector<int64_t>,
        std::vector<float>,
        std::vector<std::string>
        //Еще тензор позже и т.д.
    >;

  private:
    std::string name_;
    Value value_;

  public:
    Attribute(const std::string& name, Value value);

    const std::string& Name() const { return name_; }
    const Value& GetValue() const { return value_; }

    int64_t                         AsInt()     const;
    float                           AsFloat()   const;
    const std::string&              AsString()  const;
    const std::vector<int64_t>&     AsInts()    const;
    const std::vector<float>&       AsFloats()  const;
    const std::vector<std::string>& AsStrings() const;
    //аналогично вэрианту добавить тензор и др.
};

using AttributeMap = std::unordered_map<std::string, Attribute>;

AttributeMap ParseAttributes(const onnx::NodeProto& g_node);
const Attribute* FindAttribute(const AttributeMap& attrs, const std::string& name);

} // namespace tc

#endif // ATTRIBUTE_HPP_
