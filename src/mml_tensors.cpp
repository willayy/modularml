#include "mml_tensors.hpp"

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new tensor with the given shape and all elements set to zero.
Tensor<float> tensor_mll(const Vec<int> shape) {  // NOSONAR - function signature is correct
  auto ds = make_UPtr<Vector_mml<float>>(shape);
  auto am = make_UPtr<Arithmetic_mml>();
  return Tensor<float>(move_Ptr(ds), move_Ptr(am));
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new tensor with the given shape and data.
Tensor<float> tensor_mll(const Vec<int> shape, const Vec<float>& data) {  // NOSONAR - function signature is correct
  auto ds = make_UPtr<Vector_mml<float>>(shape, data);
  auto am = make_UPtr<Arithmetic_mml>();
  return Tensor<float>(move_Ptr(ds), move_Ptr(am));
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data The data to be set in the tensor.
/// @return A new tensor with the given shape and data.
Tensor<float> tensor_mll(const Vec<int> shape, const Vec<float> data) {  // NOSONAR - function signature is correct
  auto ds = make_UPtr<Vector_mml<float>>(shape, data);
  auto am = make_UPtr<Arithmetic_mml>();
  return Tensor<float>(move_Ptr(ds), move_Ptr(am));
}