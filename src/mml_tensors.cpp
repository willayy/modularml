#include "mml_tensors.hpp"

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new tensor with the given shape and all elements set to zero.
Tensor<float> tensor_mll(const vec<int> shape) {  // NOSONAR - function signature is correct
   auto ds = std::make_unique<Vector_mml<float>>(shape);
   auto am = std::make_unique<Arithmetic_mml>();
  return Tensor<float>(std::move(ds), std::move(am));
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new tensor with the given shape and data.
Tensor<float> tensor_mll(const vec<int> shape, const vec<float>& data) {  // NOSONAR - function signature is correct
   auto ds = std::make_unique<Vector_mml<float>>(shape, data);
   auto am = std::make_unique<Arithmetic_mml>();
  return Tensor<float>(std::move(ds), std::move(am));
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data The data to be set in the tensor.
/// @return A new tensor with the given shape and data.
Tensor<float> tensor_mll(const vec<int> shape, const vec<float> data) { // NOSONAR - function signature is correct
  auto ds = std::make_unique<Vector_mml<float>>(shape, data);
  auto am = std::make_unique<Arithmetic_mml>();
  return Tensor<float>(std::move(ds), std::move(am));
}