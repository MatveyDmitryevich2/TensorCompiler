#include "runtime/tensor.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tc::runtime {

size_t Tensor::NumElements() const {
    size_t total = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error{"runtime: dynamic shapes are not supported"};
        }
        total *= static_cast<size_t>(dim);
    }
    return total;
}

std::string ShapeToStr(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) oss << ",";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::vector<float> ReadFloatTextFile(const std::string& path) {
    std::ifstream in{path};
    if (!in.is_open()) {
        throw std::runtime_error{"runtime: unable to open input file: " + path};
    }

    std::vector<float> values;
    std::string token;
    while (in >> token) {
        for (char& c : token) {
            if (c == ',') c = ' ';
        }
        std::istringstream one{token};
        float value = 0.0f;
        while (one >> value) {
            values.push_back(value);
        }
    }
    return values;
}

void WriteFloatTextFile(const std::string& path, const Tensor& tensor) {
    const std::filesystem::path parent = std::filesystem::path{path}.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out{path};
    if (!out.is_open()) {
        throw std::runtime_error{"runtime: unable to open output file: " + path};
    }

    for (size_t i = 0; i < tensor.data.size(); ++i) {
        if (i != 0) out << ' ';
        out << std::setprecision(std::numeric_limits<float>::max_digits10) << tensor.data[i];
    }
    out << '\n';
}

} // namespace tc::runtime
