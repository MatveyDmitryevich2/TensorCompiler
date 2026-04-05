#ifndef DRIVER_OPTIONS_HPP_
#define DRIVER_OPTIONS_HPP_

#include <string>

namespace tc::driver {

struct DriverOptions {
    std::string model_path;
    std::string emit_dot_path;
    std::string emit_mlir_path;
    std::string emit_llvm_path;
    std::string emit_asm_path;

    std::string target_triple;
    std::string mcpu;
    std::string opt_level = "-O2";

    bool NeedsMlir() const {
        return !emit_mlir_path.empty() || !emit_llvm_path.empty() || !emit_asm_path.empty();
    }
};

std::string Usage(const char* argv0);
DriverOptions ParseArgs(int argc, const char* argv[]);

} // namespace tc::driver

#endif // DRIVER_OPTIONS_HPP_