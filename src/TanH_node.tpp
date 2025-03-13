#pragma once

#include "TanH_node.hpp"

template <typename T>
TanHNode<T>::TanHNode(shared_ptr<const AbstractTensor> X, shared_ptr<AbstractTensor> Y)
    : X(X), Y(Y) {}

template <typename T>
void TanHNode<T>::forward() {
  if (!areInputsFilled()) throw runtime_error("TanHNode inputs are not fully set.");
  if (!X) throw runtime_error("Failed to cast X to Tensor<T>.");
  if (!Y) throw runtime_error("Output tensor Y is not allocated.");
  Arithmetic_mml<T> arithmetic;
  arithmetic.elementwise(X, [](T x) { return tanh(x); }, Y);
}

template <typename T>
bool TanHNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void TanHNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() < 1) throw runtime_error("TanHNode expects at least one input: X.");
  auto valueX = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);
  if (!X || !valueX) throw runtime_error("Failed to cast X or input X to Tensor<T>.");
  X = *valueX;
}

template <typename T>
bool TanHNode<T>::areOutputsFilled() const {
  if (!Y) return false;
  return Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> TanHNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}