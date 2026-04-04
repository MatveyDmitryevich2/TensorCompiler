enable_testing()

add_executable(tc_tests)

target_sources(tc_tests
    PRIVATE
        graph_test.cpp
        loader_test.cpp
)

target_link_libraries(tc_tests
    PRIVATE
        tc-flags
        helpers
        graph
        onnx_loader
        onnx_proto

        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tc_tests)