add_library(mlir_backend STATIC)

target_sources(mlir_backend
    PRIVATE
        source/mlir_backend_common.cpp
        source/mlir_backend_module.cpp
        source/mlir_backend_elementwise.cpp
        source/mlir_backend_linear.cpp
        source/mlir_backend_conv.cpp
)

target_include_directories(mlir_backend
    PUBLIC
        include
)

target_link_libraries(mlir_backend
    PUBLIC
        graph
    PRIVATE
        tc-flags
        helpers
        spdlog
)
