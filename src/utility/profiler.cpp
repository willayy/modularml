#include "utility/profiler.hpp"

#include <iostream>
// IWYU pragma: no_include <__ostream/basic_ostream.h>
#include <ostream>  // IWYU pragma: keep
#include <ratio>

std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point>
    Profiler::times;

void Profiler::begin_timing(const std::string &section_name) {
  times[section_name] = std::chrono::high_resolution_clock::now();
}

void Profiler::end_timing(const std::string &section_name) {
  if (times.find(section_name) != times.end()) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - times[section_name])
                        .count();
    std::cout << "Section: [ " << section_name << " ] took " << duration
              << " ms." << std::endl;
  } else {
    std::cout << "Section: [ " << section_name << " ] not found" << std::endl;
  }
}
