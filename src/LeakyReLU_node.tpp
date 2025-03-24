#pragma once

#include "LeakyReLU_node.hpp"

template <typename T>
LeakyReLUNode<T>::LeakyReLUNode(shared_ptr<const AbstractTensor> X,
                                shared_ptr<AbstractTensor> Y, float alpha)
    : X(X), Y(Y), alpha(alpha) {}

template <typename T> void LeakyReLUNode<T>::forward() {
  if (!areInputsFilled())
    throw runtime_error("LeakyReLUNode inputs are not fully set.");
  if (!X)
    throw runtime_error("Failed to cast X to Tensor<T>.");
  if (!Y)
    throw runtime_error("Output tensor Y is not allocated.");
  leaky_relu_elementwise();
}

template <typename T> bool LeakyReLUNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void LeakyReLUNode<T>::setInputs(const array_mml<GeneralDataTypes> &inputs) {
  if (inputs.size() < 1)
    throw runtime_error("ReLUNode expects at least one input: X.");
  auto valueX = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);
  if (!X || !valueX)
    throw runtime_error("Failed to cast X or input X to Tensor<T>.");
  X = *valueX;
}

template <typename T> bool LeakyReLUNode<T>::areOutputsFilled() const {
  if (!Y)
    return false;
  return Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> LeakyReLUNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{
      GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}

template <typename T> T LeakyReLUNode<T>::leaky_relu_operation(T x) {
  if (x < 0)
    return alpha * x;
  else
    return x;
}

template <typename T> void LeakyReLUNode<T>::leaky_relu_elementwise() {
  const auto shape = X->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<int> indices(num_dimensions);
  for (uint64_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = X->get_size();

  for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply function `f` from `a` to `c`
    (*Y)[indices] = leaky_relu_operation((*X)[indices]);

    // Increment indices like a multi-dimensional counter
    for (int d = num_dimensions - 1; d >= 0; --d) {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    }
  }
}