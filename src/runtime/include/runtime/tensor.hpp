#ifndef RUNTIME_TENSOR_HPP_
#define RUNTIME_TENSOR_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace tc::runtime {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    size_t NumElements() const;
};

std::string ShapeToStr(const std::vector<int64_t>& shape);
std::vector<float> ReadFloatTextFile(const std::string& path);
void WriteFloatTextFile(const std::string& path, const Tensor& tensor);

} // namespace tc::runtime

#endif // RUNTIME_TENSOR_HPP_
