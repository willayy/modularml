#include <cassert>
#include <iostream>
#include <modularml>

#define assert_msg(name, condition)                         \
  if (!(condition)) {                                       \
    std::cerr << "Assertion failed: " << name << std::endl; \
  }                                                         \
  assert(condition);                                        \
  std::cout << name << ": " << (condition ? "Passed" : "Failed") << std::endl;


/**
 * @brief Computes the square of a given number.
 *
 * This function takes a single floating-point number as input and returns its square.
 * It is used to test elementwise operations in the modularml library.
 *
 * @param x The number to be squared.
 * @return The square of the input number.
 */
float square(float x) { return x * x; }

/**
 * @file test_mml_elementwise.cpp
 * @brief Tests for the mml_elementwise class and its functions.
 */

/**
 * @brief Main function to test the mml_elementwise class.
 * 
 * This function creates two tensors and applies an elementwise operation
 * to one of them. It then checks if the result matches the expected tensor.
 * 
 * @return int Returns 0 upon successful completion of all tests.
 */
int main() {
  mml_elementwise<float> elementwise;  // Determines what version of elementwise to use

  Tensor<float> t1 = tensor_mll({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t2 = tensor_mll({3, 3}, {1, 4, 9, 16, 25, 36, 49, 64, 81});

  // Test elementwise_apply function
  auto t3 = elementwise.apply(t1, square);
  assert_msg("Elementwise apply test", t3 == t2);

  std::cout << "All tests completed!" << std::endl;

  return 0;
}