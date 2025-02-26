#include "mml_tensors.hpp"

#include <numeric>
#include "mml_gemm.hpp"

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new tensor with the given shape and all elements set to zero.
Tensor<float> tensor_mll(const vector<int> shape) {  // NOSONAR - function signature is correct
  auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto ds = make_unique<Vector_mml<float>>(size);
  auto am = make_unique<Arithmetic_mml>();
  auto gm = make_unique<Gemm_mml>();
  return Tensor<float>(move(ds), move(am), move(gm), shape);
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new tensor with the given shape and data.
Tensor<float> tensor_mll(const vector<int> shape, const vector<float>& data) {  // NOSONAR - function signature is correct
  auto ds = make_unique<Vector_mml<float>>(data);
  auto am = make_unique<Arithmetic_mml>();
  auto gm = make_unique<Gemm_mml>();
  return Tensor<float>(move(ds), move(am), move(gm), shape);
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data The data to be set in the tensor.
/// @return A new tensor with the given shape and data.
Tensor<float> tensor_mll(const vector<int> shape, const vector<float> data) {  // NOSONAR - function signature is correct
  auto ds = make_unique<Vector_mml<float>>(data);
  auto am = make_unique<Arithmetic_mml>();
  auto gm = make_unique<Gemm_mml>();
  return Tensor<float>(move(ds), move(am), move(gm), shape);
}