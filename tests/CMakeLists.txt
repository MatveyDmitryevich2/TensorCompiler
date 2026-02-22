enable_testing()

add_executable(test)

target_sources(test
    PRIVATE
        graph_test.cpp
)

target_link_libraries(test
    PRIVATE
        tc-flags
        helpers
        graph
 
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test)
