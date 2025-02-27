#include <cassert>
#include <iostream>

#include "mml_matmul.hpp"

#define assert_msg(name, condition)                         \
  if (!(condition)) {                                       \
    std::cerr << "Assertion failed: " << name << std::endl; \
  }                                                         \
  assert(condition);                                        \
  std::cout << name << ": " << (condition ? "Passed" : "Failed") << std::endl;


int main() {
    assert_msg("Test", (1 == 1));
}