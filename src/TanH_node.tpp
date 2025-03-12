#pragma once

#include "TanH_node.hpp"

template <typename T>
TanHNode<T>::TanHNode(std::shared_ptr<AbstractTensor> X, std::shared_ptr<AbstractTensor> Y)
    : X(X), Y(Y) {}

template <typename T>
void TanHNode<T>::forward() {
  if (!areInputsFilled())
    throw std::runtime_error("TanHNode inputs are not fully set.");

  if (!X)
    throw std::runtime_error("Failed to cast X to Tensor_mml<T>.");

  if (!Y)
    throw std::runtime_error("Output tensor Y is not allocated.");

  Arithmetic_mml<T> arithmetic;
  arithmetic.elementwise_in_place(X, [](T x) { return std::tanh(x); });
  *Y = *X;
}

template <typename T>
bool TanHNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void TanHNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() < 1)
    throw std::runtime_error("TanHNode expects at least one input: X.");

  auto valueX = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);

  auto valueX_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueX);
  if (!X || !valueX_mml)
    throw std::runtime_error("Failed to cast X or input X to Tensor_mml<T>.");
  *X = *valueX_mml;
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