#include <gtest/gtest.h>

#include <modularml>

/**
 * @file test_mml_TensorFunction_Tanh.cpp
 * @brief Unit tests for the TanhFunction class.
 *
 * This file contains test cases that validate the correctness of the
 * TanhFunction class by checking its `func`, `derivative`, and `primitive`
 * methods using predefined tensors and mathematical expectations.
 *
 * The tests verify that:
 * - `func()` correctly applies tanh to each tensor element.
 * - `derivative()` correctly computes the derivative of tanh (1 - tanh^2(x)).
 * - `primitive()` correctly computes the integral of tanh (ln|cosh(x)|).
 *
 * Each test outputs a success/failure message using `assert_msg` and ensures
 * correct behavior of the TanhFunction implementation.
 */
TEST(test_mml_TensorFunction_Tanh, test_TanhFunction) {
  // replace any implementation of a TensorFunction class for a tanH function
  // here:
  tanH_mml<float> tanh_func;

  /**
   * @brief Tensor to test and compare the operations on.
   */
  shared_ptr<Tensor<float>> t1 = tensor_mml_p<float>({3, 3}, {-1.0f, 0.0f, 1.0f,
                                                              -2.0f, 2.0f, -3.0f,
                                                              3.0f, 4.0f, -4.0f});

  /**
   * @brief Expected Tensor after the tanH function is applied to each element.
   */
  shared_ptr<Tensor<float>> expected_func = tensor_mml_p<float>({3, 3}, {tanh(-1.0f), tanh(0.0f), tanh(1.0f),
                                                                         tanh(-2.0f), tanh(2.0f), tanh(-3.0f),
                                                                         tanh(3.0f), tanh(4.0f), tanh(-4.0f)});

  /**
   * @brief Expected Tensor after the derivative of the tanH function is applied
   * to each element.
   */
  shared_ptr<Tensor<float>> expected_derivative = tensor_mml_p<float>({3, 3}, {
                                                                                  1.0f - tanh(-1.0f) * tanh(-1.0f),
                                                                                  1.0f - tanh(0.0f) * tanh(0.0f),
                                                                                  1.0f - tanh(1.0f) * tanh(1.0f),
                                                                                  1.0f - tanh(-2.0f) * tanh(-2.0f),
                                                                                  1.0f - tanh(2.0f) * tanh(2.0f),
                                                                                  1.0f - tanh(-3.0f) * tanh(-3.0f),
                                                                                  1.0f - tanh(3.0f) * tanh(3.0f),
                                                                                  1.0f - tanh(4.0f) * tanh(4.0f),
                                                                                  1.0f - tanh(-4.0f) * tanh(-4.0f),
                                                                              });

  /**
   * @brief Expected Tensor after the primitive of the tanH function is applied
   * to each element.
   */
  shared_ptr<Tensor<float>> expected_primitive = tensor_mml_p<float>({3, 3}, {log(cosh(-1.0f)), log(cosh(0.0f)), log(cosh(1.0f)),
                                                                              log(cosh(-2.0f)), log(cosh(2.0f)), log(cosh(-3.0f)),
                                                                              log(cosh(3.0f)), log(cosh(4.0f)), log(cosh(-4.0f))});

  auto func_result = tanh_func.func(t1);
  auto derivative_result = tanh_func.derivative(t1);
  auto primitive_result = tanh_func.primitive(t1);

  // Test applying tanH
  ASSERT_TRUE(tensors_are_close((*func_result), (*expected_func)));
  // Test applying the derivative of tanH
  ASSERT_TRUE(tensors_are_close((*derivative_result), (*expected_derivative)));
  // Test applying the primitive of tanH
  ASSERT_TRUE(tensors_are_close((*primitive_result), (*expected_primitive)));
}

TEST(test_mml_TensorFunction_Tanh, test_TanhFunction_double) {
  // Define the tanH function class to test
  tanH_mml<double> tanh_func;

  /**
   * @brief Tensor to test and compare the operations on.
   */
  shared_ptr<Tensor<double>> t1 = tensor_mml_p<double>({3, 3}, {-1.0, 0.0, 1.0,
                                                                -2.0, 2.0, -3.0,
                                                                3.0, 4.0, -4.0});

  /**
   * @brief Expected Tensor after the tanH function is applied to each element.
   */
  shared_ptr<Tensor<double>> expected_func = tensor_mml_p<double>({3, 3}, {tanh(-1.0), tanh(0.0), tanh(1.0),
                                                                           tanh(-2.0), tanh(2.0), tanh(-3.0),
                                                                           tanh(3.0), tanh(4.0), tanh(-4.0)});

  /**
   * @brief Expected Tensor after the derivative of the tanH function is applied
   * to each element.
   */
  shared_ptr<Tensor<double>> expected_derivative = tensor_mml_p<double>({3, 3}, {1.0 - tanh(-1.0) * tanh(-1.0), 1.0 - tanh(0.0) * tanh(0.0), 1.0 - tanh(1.0) * tanh(1.0),
                                                                                 1.0 - tanh(-2.0) * tanh(-2.0), 1.0 - tanh(2.0) * tanh(2.0), 1.0 - tanh(-3.0) * tanh(-3.0),
                                                                                 1.0 - tanh(3.0) * tanh(3.0), 1.0 - tanh(4.0) * tanh(4.0), 1.0 - tanh(-4.0) * tanh(-4.0)});

  /**
   * @brief Expected Tensor after the primitive of the tanH function is applied
   * to each element.
   */
  shared_ptr<Tensor<double>> expected_primitive = tensor_mml_p<double>({3, 3}, {log(cosh(-1.0)), log(cosh(0.0)), log(cosh(1.0)),
                                                                                log(cosh(-2.0)), log(cosh(2.0)), log(cosh(-3.0)),
                                                                                log(cosh(3.0)), log(cosh(4.0)), log(cosh(-4.0))});

  auto func_result = tanh_func.func(t1);
  auto derivative_result = tanh_func.derivative(t1);
  auto primitive_result = tanh_func.primitive(t1);

  // Test applying tanH
  ASSERT_TRUE(tensors_are_close((*func_result), (*expected_func)));
  // Test applying the derivative of tanH
  ASSERT_TRUE(tensors_are_close((*derivative_result), (*expected_derivative)));
  // Test applying the primitive of tanH
  ASSERT_TRUE(tensors_are_close((*primitive_result), (*expected_primitive)));
}
