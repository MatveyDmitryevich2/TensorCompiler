add_library(runtime STATIC)

target_sources(runtime
    PRIVATE
        source/interpreter.cpp
        source/tensor.cpp
)

target_include_directories(runtime
    PUBLIC
        include
)

target_link_libraries(runtime
    PRIVATE
        tc-flags
        spdlog
    PUBLIC
        graph
)
