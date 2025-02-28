#pragma once

#include <iostream>
#include <cmath>
#include "tensor.hpp"

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
static bool tensors_are_close(Tensor<float>& t1, Tensor<float>& t2, float tolerance = 0.01f) {
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