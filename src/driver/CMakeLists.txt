add_library(driver STATIC)

target_sources(driver
    PRIVATE
        source/driver_options.cpp
        source/tool_runner.cpp
)

target_include_directories(driver
    PUBLIC
        include
)

target_link_libraries(driver
    PRIVATE
        tc-flags
        spdlog
)
