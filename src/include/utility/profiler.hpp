#pragma once

#include "globals.hpp"

/// @brief A utility for profiling sections of code by measuring execution time.
class Profiler {
public:

    /**
     * @brief Starts the timer for a section
     * 
     * @param section_name The name of the section that is being profiled. Used to keep track of different sections.
     */
    static void begin_timing(const string& section_name);

    /**
     * @brief Ends the timer for a section and prints the elapsed time to standard output
     * 
     * Calculates the time that has passed since using begin_timing 
     * 
     * @param section_name The name of the section that is being profiled. Used to keep track of different sections.
     */
    static void end_timing(const string& section_name);

private:

    /**
     * @brief A static map that stores the start time for each section
     * 
     * The key is the name of the section and the value is the time when begin_timing was first called for that section
     */
    static std::unordered_map<string, std::chrono::high_resolution_clock::time_point> times;
};

/**
 * @example
 * 
 * Example usage:
 * 
 * ```cpp
 * #include <iostream>
 * #include "Profiler.h"
 * 
 * int main() {
 *     Profiler::begin_timing("ExampleSection");
 *     
 *     // Simulate some workload
 *     for (volatile int i = 0; i < 1000000; ++i);
 *     
 *     Profiler::end_timing("ExampleSection");
 *     return 0;
 * }
 * ```
 * 
 * Output:
 * ```
 * ExampleSection took 3.45 ms
 * ```
 */


