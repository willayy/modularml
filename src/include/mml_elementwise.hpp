#pragma once

#include "tensor.hpp"

/**
 * @brief Applies a given function element-wise to a tensor.
 *
 * This function iterates over each element in the given tensor and applies
 * the provided function to each element, modifying the tensor in place.
 *
 * @param t The tensor to which the function will be applied.
 * @param f A pointer to the function that will be applied to each element of
 * the tensor. The function should take a float as input and return a float.
 * @return A reference to the modified tensor.
 */
Tensor<float>& elementwise_apply(Tensor<float>& t, float (*f)(float)) {
  // This function can be made way more efficent by the use of multi-threading
  // I intend on making that an improvement in the future
  for (int i = 0; i < t.get_shape()[0]; i++) {
    for (int j = 0; j < t.get_shape()[1]; j++) {
      t[{i, j}] = f(t[{i, j}]);
    }
  }
  return t;
}
