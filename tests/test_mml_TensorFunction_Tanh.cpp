#include <cassert>
#include <iostream>
#include <modularml>
#include <numeric>
#include <random>
#include <vector>

#include "mml_TensorFunction_TanH.hpp"

#define assert_msg(name, condition)                         \
  if (!(condition)) {                                       \
    std::cerr << "Assertion failed: " << name << std::endl; \
  }                                                         \
  assert(condition);                                        \
  std::cout << name << ": " << (condition ? "Passed" : "Failed") << std::endl;

Vec<float> random_vec_f(int size, float low, float max, int seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(low, max);
  Vec<float> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = dist(gen);
  }
  return v;
}

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
int main() {
  // replace any implementation of a TensorFunction class for a tanH function
  // here:
  mml_TensorFunction_Tanh tanh_func;

  Tensor<float> t1 = tensor_mll({3, 3}, {-1, 0, 1, -2, 2, -3, 3, 4, -4});

  Tensor<float> expected_func = tensor_mll(
      {3, 3},
      {static_cast<float>(std::tanh(-1)), static_cast<float>(std::tanh(0)),
       static_cast<float>(std::tanh(1)), static_cast<float>(std::tanh(-2)),
       static_cast<float>(std::tanh(2)), static_cast<float>(std::tanh(-3)),
       static_cast<float>(std::tanh(3)), static_cast<float>(std::tanh(4)),
       static_cast<float>(std::tanh(-4))});

  Tensor<float> expected_derivative = tensor_mll(
      {3, 3}, {
                  static_cast<float>(1 - std::tanh(-1) * std::tanh(-1)),
                  static_cast<float>(1 - std::tanh(0) * std::tanh(0)),
                  static_cast<float>(1 - std::tanh(1) * std::tanh(1)),
                  static_cast<float>(1 - std::tanh(-2) * std::tanh(-2)),
                  static_cast<float>(1 - std::tanh(2) * std::tanh(2)),
                  static_cast<float>(1 - std::tanh(-3) * std::tanh(-3)),
                  static_cast<float>(1 - std::tanh(3) * std::tanh(3)),
                  static_cast<float>(1 - std::tanh(4) * std::tanh(4)),
                  static_cast<float>(1 - std::tanh(-4) * std::tanh(-4)),
              });

  Tensor<float> expected_primitive =
      tensor_mll({3, 3}, {static_cast<float>(std::log(std::cosh(-1))),
                          static_cast<float>(std::log(std::cosh(0))),
                          static_cast<float>(std::log(std::cosh(1))),
                          static_cast<float>(std::log(std::cosh(-2))),
                          static_cast<float>(std::log(std::cosh(2))),
                          static_cast<float>(std::log(std::cosh(-3))),
                          static_cast<float>(std::log(std::cosh(3))),
                          static_cast<float>(std::log(std::cosh(4))),
                          static_cast<float>(std::log(std::cosh(-4)))});

  assert_msg("TanhFunction func test", tanh_func.func(t1) == expected_func);
  assert_msg("TanhFunction derivative test",
             tanh_func.derivative(t1) == expected_derivative);
  assert_msg("TanhFunction primitive test",
             tanh_func.primitive(t1) == expected_primitive);

  std::cout << "All TanhFunction tests completed!" << std::endl;
  return 0;
}
