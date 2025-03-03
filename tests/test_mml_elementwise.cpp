#include <gtest/gtest.h>

#include <iostream>
#include <modularml>

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
TEST(test_mml_elementwise, test_elementwise_apply) {
  mml_elementwise<float> elementwise;  // Determines what version of elementwise to use

  Tensor<float> t1 = tensor_mml<float>({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  Tensor<float> t2 = tensor_mml<float>({3, 3}, {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f, 81.0f});

  // Test elementwise_apply function
  auto t3 = elementwise.apply(t1, square);
  ASSERT_EQ(t3, t2);
}