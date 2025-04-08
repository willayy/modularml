#include <gtest/gtest.h>

#include <thread>
#include "utility/profiler.hpp"

TEST(test_profiler, test_begin_end_timing) {
    using namespace std::chrono_literals;

    // Just check that it executes
    Profiler::begin_timing("Test1");
    Profiler::end_timing("Test1");

    // Check that it prints correctly to stdout
    Profiler::begin_timing("Test2");

    testing::internal::CaptureStdout();

    Profiler::end_timing("Test2");

    string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("Section: [ Test2 ] took") != string::npos);
}