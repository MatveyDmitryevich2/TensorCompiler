include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
    spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog
        GIT_TAG v1.15.0
        GIT_SHALLOW true
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    onnx
        GIT_REPOSITORY https://github.com/onnx/onnx
        GIT_TAG v1.20.1
        GIT_SHALLOW true
)
FetchContent_MakeAvailable(onnx)

FetchContent_Declare(
    googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
