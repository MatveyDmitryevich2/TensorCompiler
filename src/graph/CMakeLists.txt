add_library(graph STATIC)

target_sources(graph
    PRIVATE
        source/graph.cpp
)

target_include_directories(graph
    PUBLIC
        include
)

target_link_libraries(graph
    PRIVATE
        tc-flags

        spdlog
)
