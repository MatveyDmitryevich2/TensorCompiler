cmake_minimum_required(VERSION 3.21)

project(tc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(flags)
add_subdirectory(third_party)

add_subdirectory(src/helpers)
add_subdirectory(src/graph)
add_subdirectory(src/onnx_loader)
add_subdirectory(src/driver)
add_subdirectory(src/mlir_backend)

add_executable(tc.x)

target_sources(tc.x
    PRIVATE
        src/main.cpp
)

target_include_directories(tc.x
    PRIVATE
        include/
)

target_link_libraries(tc.x
    PRIVATE
        tc-flags
        helpers
        graph
        onnx_loader
        driver
        mlir_backend

        spdlog
)

enable_testing()
add_subdirectory(tests)