#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"
#include "mml_tensor.hpp"

/**
 * @brief Compares two tensors element-wise to check if they are close within a specified tolerance.
 *
 * This function iterates over each element of the given tensors and checks if the absolute difference
 * between corresponding elements is within the specified tolerance. If any pair of elements differ by
 * more than the tolerance, the function returns false. Otherwise, it returns true.
 *
 * @tparam T The data type of the tensor elements (must be an arithmetic type).
 * @param t1 The first tensor to compare.
 * @param t2 The second tensor to compare.
 * @param tolerance The maximum allowed difference between corresponding elements of the tensors seen as a percentage. Default is 0.01.
 * @return true if all corresponding elements of the tensors are within the specified tolerance, false otherwise.
 */
template <typename T>
static bool tensors_are_close(Tensor<T>& t1, Tensor<T>& t2, T tolerance = T(0.01)) {
  static_assert(std::is_arithmetic_v<T>, "Tensor type must be an arithmetic type (int, float, double, etc.).");

  if (t1.get_shape() != t2.get_shape()) {
    std::cerr << "Error: Tensors have different shapes and cannot be compared!" << std::endl;
    return false;
  }

  for (int i = 0; i < t1.get_size(); i++) {
    T diff = std::abs(t1[i] - t2[i]);
    T tolerance_limit = std::abs(tolerance * (t2[i]));

    if (diff > tolerance_limit) {
      std::cerr << "Difference of " << diff << " found at (" << i << ") which is too large." << std::endl;
      std::cerr << "Tolerance limit is " << tolerance_limit << std::endl;
      return false;
    }
  }

  return true;
}

#define GENERATE_RANDOM_ARRAY_INTEGRAL(T) (std::is_integral_v<T>, "Random array generation (integral) requires an integral type (int, long, etc.).");
template <typename T>
static array_mml<T> generate_random_array_mml_integral(uint64_t lo_sz = 1, uint64_t hi_sz = 5, T lo_v = 1, T hi_v = 10) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution size_dist(lo_sz, hi_sz);
  uint64_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_int_distribution<T> int_dist(lo_v, hi_v);
  for (int i = 0; i < n; i++) {
    arr[i] = int_dist(gen);
  }
  return arr;
}

#define GENERATE_RANDOM_ARRAY_REAL(T) (std::is_floating_point_v<T>, "Random array generation (real) requires a floating-point type (float, double, etc.).");
template <typename T>
static array_mml<T> generate_random_array_mml_real(uint64_t lo_sz = 1, uint64_t hi_sz = 5, T lo_v = 1, T hi_v = 100) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution size_dist(lo_sz, hi_sz);
  uint64_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_real_distribution<T> real_dist(lo_v, hi_v);
  for (int i = 0; i < n; i++) {
    arr[i] = real_dist(gen);
  }
  return arr;
}

/**
 * @brief Generates a random tensor with the specified shape and value range.
 *
 * This function creates a tensor with random values within the specified range.
 * The type of the tensor elements must be an arithmetic type.
 *
 * @tparam T The data type of the tensor elements (must be an arithmetic type).
 * @param shape The shape of the tensor to generate.
 * @param lo_v The lower bound of the random values.
 * @param hi_v The upper bound of the random values.
 * @return A tensor with random values within the specified range.
 */
#define GENERATE_RANDOM_TENSOR(T) (std::is_arithmetic_v<T>, "Random Tensor generation requires an arithmetic type (int, float, double, etc.).");
template <typename T>
static auto generate_random_tensor(const array_mml<int>& shape, T lo_v = T(0), T hi_v = T(1)) {
  static_assert(std::is_arithmetic_v<T>, "Tensor type must be an arithmetic type (int, float, double, etc.).");
  Tensor_mml<T> tensor(shape);
  std::random_device rd;
  std::mt19937 gen(rd());

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(lo_v, hi_v);
    for (size_t i = 0; i < tensor.get_size(); i++) {
      tensor[i] = dist(gen);
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(lo_v, hi_v);
    for (size_t i = 0; i < tensor.get_size(); i++) {
      tensor.operator[](i) = dist(gen);
    }
  }

  return move(tensor);
}