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
 * @brief Compares two tensors element-wise to check if they are close within a specified tolerance.
 *
 * This function iterates over each element of the given tensors and checks if the absolute difference
 * between corresponding elements is within the specified tolerance. If any pair of elements differ by
 * more than the tolerance, the function returns false. Otherwise, it returns true.
 *
 * @param t1 The first tensor to compare.
 * @param t2 The second tensor to compare.
 * @param tolerance The maximum allowed difference between corresponding elements of the tensors seen as a percentage. Default is 0.01f.
 * @return true if all corresponding elements of the tensors are within the specified tolerance, false otherwise.
 */
bool tensors_are_close(Tensor<float>& t1, Tensor<float>& t2, float tolerance = 0.01f) {
  for (int i = 0; i < t1.get_shape()[0]; i++) {
    for (int j = 0; j < t1.get_shape()[1]; j++) {
      if (std::abs(t1[{i, j}] - t2[{i, j}]) > std::abs(tolerance * t2[{i, j}])) {
        std::cerr << "Difference of " << std::abs(t1[{i, j}] - t2[{i, j}]) << " found at (" << i << ", " << j << ") which is too large." << std::endl;
        std::cerr << "Tolerance limit is " << std::abs(tolerance * t2[{i, j}]) << std::endl;
        return false;
      }
    }
  }
  return true;
}

/**
 * @file test_mml_TensorFunction_Swish.cpp
 * @brief Unit tests for the SwishFunction class.
 *
 * This file contains test cases that validate the correctness of the
 * SwishFunction class by checking its `func`, `derivative`, and `primitive`
 * methods using predefined tensors. Unlike some of the other tests for
 * activation function this test simply makes sure that the return from the
 * functions are within an expected range of the correct value.
 *
 * The tests verify that:
 * - `func()` correctly applies Swish to each tensor element.
 * - `derivative()` correctly computes and applies the derivative of Swish.
 *
 * - `primitive()` correctly computes and applies the approximation primitive of Swish.
 *
 * Each test outputs a success/failure message using `assert_msg` and ensures
 * correct behavior of the ReLUFunction implementation.
 */
int main() {
  // Define the Swish function class to test
  mml_TensorFunction_Swish Swish_func;

  /**
   * @brief Tensor to test and compare the operations on.
   */
  Tensor<float> t1 = tensor_mll({3, 3}, {-1, 0, 1, -2, 2, -3, 3, 4, -4});

  /**
   * @brief Expected Tensor after the Swish function is applied to each element.
   */
  Tensor<float> expected_func = tensor_mll({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});

  /**
   * @brief Expected Tensor after the derivative of the Swish function is applied
   * to each element.
   */
  Tensor<float> expected_derivative = tensor_mll({3, 3}, {0.0723f, 0.5000f, 0.9277f, -0.0908f, 1.0908f, -0.0881f, 1.0881f, 1.0527f, -0.0527f});

  /**
   * @brief Expected Tensor after the approximation of the primitive of the Swish function is applied
   * to each element.
   */
  Tensor<float> expected_primitive = tensor_mll({3, 3}, {1.0443f, 0.6931f, 1.0443f, 1.8885f, 1.8885f, 2.9063f, 2.9063f, 3.9462f, 3.9462f});

  // Test applying Swish
  auto func_result = Swish_func.func(t1);
  auto derivative_result = Swish_func.derivative(t1);
  auto primitive_result = Swish_func.primitive(t1);

  // Checks if the results are within 99% of the expected values
  assert_msg("SwishFunction func test", tensors_are_close(func_result, expected_func));
  // Test applying the derivative of Swish
  assert_msg("SwishFunction derivative test", tensors_are_close(derivative_result, expected_derivative));
  // Test applying the approximation of the primitive of Swish
  assert_msg("SwishFunction primitive test", tensors_are_close(primitive_result, expected_primitive));

  std::cout << "All SwishFunction tests passed!" << std::endl;
  return 0;
}