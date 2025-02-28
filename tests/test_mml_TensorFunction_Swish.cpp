#include <gtest/gtest.h>

#include <modularml>

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
TEST(test_mml_TensorFunction_Swish, test_SwishFunction) {
  // Define the Swish function class to test
  mml_TensorFunction_Swish Swish_func;

  /**
   * @brief Tensor to test and compare the operations on.
   */
  Tensor<float> t1 = tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  /**
   * @brief Expected Tensor after the Swish function is applied to each element.
   */
  Tensor<float> expected_func = tensor_mml<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});

  /**
   * @brief Expected Tensor after the derivative of the Swish function is applied
   * to each element.
   */
  Tensor<float> expected_derivative = tensor_mml<float>({3, 3}, {0.0723f, 0.5000f, 0.9277f, -0.0908f, 1.0908f, -0.0881f, 1.0881f, 1.0527f, -0.0527f});

  /**
   * @brief Expected Tensor after the approximation of the primitive of the Swish function is applied
   * to each element.
   */
  Tensor<float> expected_primitive = tensor_mml<float>({3, 3}, {1.0443f, 0.6931f, 1.0443f, 1.8885f, 1.8885f, 2.9063f, 2.9063f, 3.9462f, 3.9462f});

  // Test applying Swish
  auto func_result = Swish_func.func(t1);
  auto derivative_result = Swish_func.derivative(t1);
  auto primitive_result = Swish_func.primitive(t1);

  // Checks if the results are within 99% of the expected values
  ASSERT_TRUE(tensors_are_close(func_result, expected_func));
  // Test applying the derivative of Swish
  ASSERT_TRUE(tensors_are_close(derivative_result, expected_derivative));
  // Test applying the approximation of the primitive of Swish
  ASSERT_TRUE(tensors_are_close(primitive_result, expected_primitive));
}