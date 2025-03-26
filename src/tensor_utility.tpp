#pragma once

#include "tensor_utility.hpp"

template <typename T>
bool tensors_are_close(Tensor<T> &t1, Tensor<T> &t2, T tolerance) {
  static_assert(
      std::is_arithmetic_v<T>,
      "Tensor type must be an arithmetic type (int, float, double, etc.).");

  if (t1.get_shape() != t2.get_shape()) {
    std::cerr << "Error: Tensors have different shapes and cannot be compared!"
              << std::endl;
    return false;
  }

  for (uli i = 0; i < t1.get_size(); i++) {
    T diff = std::abs(t1[i] - t2[i]);
    T tolerance_limit = std::max(0.00001f, std::abs(tolerance * (t2[i])));

    if (diff > tolerance_limit) {
      std::cerr << "Difference of " << diff << " found at (" << i
                << ") which is too large." << std::endl;
      std::cerr << "Tolerance limit is " << tolerance_limit << std::endl;
      return false;
    }
  }

  return true;
}