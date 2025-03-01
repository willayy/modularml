#include <gtest/gtest.h>

#include <modularml>

/**
 * @file test_mml_TensorFunction_ReLU.cpp
 * @brief Unit tests for the ReLUFunction class.
 *
 * This file contains test cases that validate the correctness of the
 * ReLUFunction class by checking its `func`, `derivative`, and `primitive`
 * methods using predefined tensors.
 *
 * The tests verify that:
 * - `func()` correctly applies ReLU to each tensor element (max(0, x)).
 * - `derivative()` correctly computes the derivative of ReLU (1 for x > 0, else
 * 0).
 * - `primitive()` correctly computes the integral of ReLU (0.5 * max(0, x)^2).
 *
 * Each test outputs a success/failure message using `assert_msg` and ensures
 * correct behavior of the ReLUFunction implementation.
 */
TEST(test_mml_TensorFunction_ReLU, test_ReLUFunction) {
  // Define the ReLU function class to test
  ReLU_mml<float> ReLU_func;
  /**
   * @brief Tensor to test and compare the operations on.
   */
  Tensor<float> t1 = tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  Tensor<float> expected_func = tensor_mml<float>(
      {3, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f});

  /**
   * @brief Expected Tensor after the derivative of the ReLU function is applied
   * to each element.
   */
  Tensor<float> expected_derivative = tensor_mml<float>(
      {3, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f});

  /**
   * @brief Expected Tensor after the primitive of the ReLU function is applied
   * to each element.
   */
  Tensor<float> expected_primitive = tensor_mml<float>(
      {3, 3}, {0.0f, 0.0f, 0.5f, 0.0f, 2.0f, 0.0f, 4.5f, 8.0f, 0.0f});

  // Test applying ReLU
  ASSERT_EQ(ReLU_func.func(t1), expected_func);
  // Test applying the derivative of ReLU
  ASSERT_EQ(ReLU_func.derivative(t1), expected_derivative);
  // Test applying the primitive of ReLU
  ASSERT_EQ(ReLU_func.primitive(t1), expected_primitive);
}

TEST(test_mml_TensorFunction_ReLU, test_ReLUFunction_int) {
    // Define the ReLU function class to test
    ReLU_mml<int> ReLU_func;
  
    /**
     * @brief Tensor to test and compare the operations on.
     */
    Tensor<int> t1 = tensor_mml<int>({3, 3}, {-1, 0, 1, -2, 2, -3, 3, 4, -4});
  
    /**
     * @brief Expected Tensor after the ReLU function is applied to each element.
     */
    Tensor<int> expected_func = tensor_mml<int>(
        {3, 3}, {0, 0, 1, 0, 2, 0, 3, 4, 0});
  
    /**
     * @brief Expected Tensor after the derivative of the ReLU function is applied
     * to each element.
     */
    Tensor<int> expected_derivative = tensor_mml<int>(
        {3, 3}, {0, 0, 1, 0, 1, 0, 1, 1, 0});
  
    /**
     * @brief Expected Tensor after the primitive of the ReLU function is applied
     * to each element.
     *
     * NOTE: Since `primitive(x) = (x * x) / 2`, integer division rounds down.
     */
    Tensor<int> expected_primitive = tensor_mml<int>(
        {3, 3}, {0, 0, 0, 0, 2, 0, 4, 8, 0});  // Integer division rounds down
  
    // Test applying ReLU
    ASSERT_EQ(ReLU_func.func(t1), expected_func);
    // Test applying the derivative of ReLU
    ASSERT_EQ(ReLU_func.derivative(t1), expected_derivative);
    // Test applying the primitive of ReLU
    ASSERT_EQ(ReLU_func.primitive(t1), expected_primitive);
  }