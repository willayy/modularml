#pragma once

#include "globals.hpp"
#include "mml_vector.hpp"
#include "tensor.hpp"

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new tensor with the given shape and all elements set to zero.
template <typename T>
shared_ptr<Tensor<T>> tensor_mml(const initializer_list<int> shape) {  // NOSONAR - function signature is correct
  auto size = accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
  auto ds = make_shared<Vector_mml<T>>(size);
  return make_shared<Tensor<T>>(move(ds), shape);
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new tensor with the given shape and data.
template <typename T>
shared_ptr<Tensor<T>> tensor_mml(const initializer_list<int> &shape, const initializer_list<T> &data) {  // NOSONAR - function signature is correct
  auto ds = make_shared<Vector_mml<T>>(data);
  return make_shared<Tensor<T>>(move(ds), shape);
}