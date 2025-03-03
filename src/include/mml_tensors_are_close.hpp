#pragma once

#include <iostream>
#include <cmath>
#include <type_traits>
#include "tensor.hpp"

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
  static_assert(std::is_arithmetic<T>::value, "Tensor type must be an arithmetic type (int, float, double, etc.).");

  if (t1.get_shape() != t2.get_shape()) {
    std::cerr << "Error: Tensors have different shapes and cannot be compared!" << std::endl;
    return false;
  }

  for (int i = 0; i < t1.get_shape()[0]; i++) {
    for (int j = 0; j < t1.get_shape()[1]; j++) {
      T diff = std::abs(t1[{i, j}] - t2[{i, j}]);
      T tolerance_limit = std::abs(tolerance * t2[{i, j}]);

      if (diff > tolerance_limit) {
        std::cerr << "Difference of " << diff << " found at (" << i << ", " << j << ") which is too large." << std::endl;
        std::cerr << "Tolerance limit is " << tolerance_limit << std::endl;
        return false;
      }
    }
  }
  return true;
}
