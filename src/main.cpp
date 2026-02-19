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
            "   usage: " << argv[0]  << " <model_path> [--dot out.dot] [--svg out.svg]\n";

        return EXIT_FAILURE;
    }

    try {
        std::string dot_path;
        std::string svg_path;

        for (int i = 2; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--dot" && i + 1 < argc) {
                dot_path = argv[++i];
            } else if (a == "--svg" && i + 1 < argc) {
                svg_path = argv[++i];
            } else {
                throw std::runtime_error{"Unknown arg: " + a};
            }
        }

        tc::OnnxLoader ld{};
        tc::Graph graph = ld.Load(argv[1]);

        if (dot_path.empty() && svg_path.empty()) {
            for (auto&& n: graph) {
                std::cout << n->ToStr() << "\n";
            }
            return EXIT_SUCCESS;
        }

        tc::DotOptions opt;
        std::string dot = graph.ToDot(opt);

        if (!dot_path.empty()) {
            std::ofstream f(dot_path);
            if (!f.is_open()) {
                throw std::runtime_error{"Unable to open file: " + dot_path};
            }
            f << dot;
            spdlog::info("Wrote DOT: {}", dot_path);
        }

        if (!svg_path.empty()) {
            std::string tmp_dot = svg_path + ".dot";
            {
                std::ofstream f(tmp_dot);
                if (!f.is_open()) {
                    throw std::runtime_error{"Unable to open file: " + tmp_dot};
                }
                f << dot;
            }

            std::string cmd = "dot -Tsvg \"" + tmp_dot + "\" -o \"" + svg_path + "\"";
            int rc = std::system(cmd.c_str());
            if (rc != 0) {
                throw std::runtime_error{
                    "Graphviz 'dot' failed (is graphviz installed?). Command: " + cmd
                };
            }
            spdlog::info("Wrote SVG: {}", svg_path);
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Unknown exception\n";
    }
}
