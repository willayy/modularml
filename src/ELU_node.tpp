#pragma once

#include "ELU_node.hpp"

template <typename T>
ELUNode<T>::ELUNode(shared_ptr<AbstractTensor> X, shared_ptr<AbstractTensor> Y,
                    float alpha)
    : X(X), Y(Y), alpha(alpha){};

template <typename T> void ELUNode<T>::forward() {
  if (!areInputsFilled())
    throw runtime_error("ELUNode inputs are not fully set.");
  if (!X)
    throw runtime_error("Failed to cast X to Tensor<T>.");
  if (!Y)
    throw runtime_error("Output tensor Y is not allocated.");
  elu_elementwise();
}

template <typename T> bool ELUNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void ELUNode<T>::setInputs(const array_mml<GeneralDataTypes> &inputs) {
  if (inputs.size() < 1)
    throw runtime_error("SigmoidNode expects at least one input: X.");
  auto valueX = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);
  if (!X || !valueX)
    throw runtime_error("Failed to cast X or input X to Tensor<T>.");
  X = *valueX;
}

template <typename T> bool ELUNode<T>::areOutputsFilled() const {
  if (!Y)
    return false;
  return Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> ELUNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{
      GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}

template <typename T> T ELUNode<T>::elu_operation(T x) {
  return x < 0 ? alpha * (exp(x) - 1) : x;
}

template <typename T> void ELUNode<T>::elu_elementwise() {
  const auto shape = X->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<uli> indices(num_dimensions);
  for (uli i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = X->get_size();

  for (uli linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply function `f` from `a` to `c`
    (*Y)[indices] = elu_operation((*X)[indices]);

    // Increment indices like a multi-dimensional counter
    for (uli d = num_dimensions - 1; d >= 0; --d) {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    }
  }
}