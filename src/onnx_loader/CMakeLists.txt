add_library(onnx_loader STATIC)

target_sources(onnx_loader
    PRIVATE
        source/onnx_loader.cpp
)

target_include_directories(onnx_loader
    PUBLIC
        include
)

target_link_libraries(onnx_loader
    PRIVATE
        tc-flags
        graph

        spdlog
)
