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

