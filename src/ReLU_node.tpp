#pragma once

#include "ReLU_node.hpp"

template <typename T>
ReLUNode<T>::ReLUNode(std::shared_ptr<const AbstractTensor> X, std::shared_ptr<AbstractTensor> Y)
    : X(X), Y(Y) {}

template <typename T>
void ReLUNode<T>::forward() {
  if (!areInputsFilled()) throw runtime_error("ReLUNode inputs are not fully set.");
  if (!X) throw runtime_error("Failed to cast X to Tensor<T>.");
  if (!Y) throw runtime_error("Output tensor Y is not allocated.");
  Arithmetic_mml<T> arithmetic;
  arithmetic.elementwise(X, [](T x) { return x > 0 ? x : 0; }, Y);
}

template <typename T>
bool ReLUNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void ReLUNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() < 1) throw runtime_error("ReLUNode expects at least one input: X.");
  auto valueX = std::get<shared_ptr<AbstractTensor>>(inputs[0]);
  if (!X || !valueX) throw runtime_error("Failed to cast X or input X to Tensor<T>.");
  X = std::const_pointer_cast<AbstractTensor>(valueX);
}

template <typename T>
bool ReLUNode<T>::areOutputsFilled() const {
  if (!Y) return false;
  return Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> ReLUNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}