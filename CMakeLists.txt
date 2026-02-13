cmake_minimum_required(VERSION 3.21)

project(tc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # to generate compile_commands.json

add_subdirectory(flags)
add_subdirectory(third_party)

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
        spdlog
)

# enable_testing()
# add_subdirectory(tests)
