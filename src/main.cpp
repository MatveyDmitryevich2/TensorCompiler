#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "graph/graph.hpp"
#include "onnx_loader/onnx_loader.hpp"

int main(int argc, const char* argv[]) {
    auto logger = spdlog::basic_logger_mt("tc", "tc.log", true);
    spdlog::set_default_logger(logger);

    spdlog::set_pattern("[%l] %v"); // remove time and name(%n) from log

#if defined (NDEBUG)
    spdlog::set_level(spdlog::level::info);
#else // NDEBUG
    spdlog::flush_on(spdlog::level::trace);
    spdlog::set_level(spdlog::level::trace);
#endif // NDEBUG

    // log argv
    for (int i = 0; i < argc; i++) {
        spdlog::info("argv[{}]: {}", i, argv[i]);
    }

    if (argc < 2) {
        std::cerr << 
            "ERROR: not enough args\n" 
            "   usage: " << argv[0]  << " <model_path>\n";

        return EXIT_FAILURE;
    }

    try {
        tc::OnnxLoader ld{};
        tc::Graph graph = ld.Load(argv[1]);
        for (auto&& n: graph) {
            std::cout << n->ToStr() << "\n";
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Unknown exception\n";
    }
}
