#pragma once

#include "Swish_node.hpp"

template <typename T>
SwishNode<T>::SwishNode(std::shared_ptr<AbstractTensor> X, std::shared_ptr<AbstractTensor> Y)
    : X(X), Y(Y) {}

template <typename T>
void SwishNode<T>::forward() {
  if (!areInputsFilled())
    throw std::runtime_error("SwishNode inputs are not fully set.");

  if (!X)
    throw std::runtime_error("Failed to cast X to Tensor_mml<T>.");

  if (!Y)
    throw std::runtime_error("Output tensor Y is not allocated.");

  Arithmetic_mml<T> arithmetic;
  arithmetic.elementwise_in_place(X, [](T x) {
    T sigmoid_x = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    return x * sigmoid_x;
  });
  *Y = *X;
}

template <typename T>
bool SwishNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void SwishNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() < 1)
    throw std::runtime_error("SwishNode expects at least one input: X.");

  auto valueX = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);

  auto valueX_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueX);
  if (!X || !valueX_mml)
    throw std::runtime_error("Failed to cast X or input X to Tensor_mml<T>.");
  *X = *valueX_mml;
}

template <typename T>
bool SwishNode<T>::areOutputsFilled() const {
  if (!Y) return false;
  return Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> SwishNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}