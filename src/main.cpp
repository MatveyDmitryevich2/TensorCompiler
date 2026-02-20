#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <stdexcept>

#include "graph/graph.hpp"
#include "onnx_loader/onnx_loader.hpp"

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
            "   usage: " << argv[0]  << " <model_path> [<dot_path>]\n";

        return EXIT_FAILURE;
    }

    try {
        tc::OnnxLoader ld{};
        tc::Graph graph = ld.Load(argv[1]);

        tc::DotOptions opt;
        std::string dot = graph.ToDot(opt);

        std::string dot_path;
        if (argc >= 3) {
            dot_path = argv[2];
        }

        if (!dot_path.empty()) {
            std::ofstream dot_file{dot_path};
            if (!dot_file.is_open()) {
                throw std::runtime_error{"Unable to open file: " + dot_path};
            }
            dot_file << dot;
            spdlog::info("Wrote DOT: {}", dot_path);
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Unknown exception\n";
    }
}
