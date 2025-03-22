#pragma once

#include "Gelu_node.hpp"

template <typename T>
GeluNode<T>::GeluNode(shared_ptr<const AbstractTensor> X,
                      shared_ptr<AbstractTensor> Y, string approximate)
    : X(X), Y(Y) {
  if (approximate == "none" || approximate == "tanh") {
    this->approximate = approximate;
  } else {
    throw invalid_argument("Invalid value for argument approximate.");
  }
}

template <typename T> void GeluNode<T>::forward() {
  if (!areInputsFilled())
    throw runtime_error("GeluNode inputs are not fully set.");
  if (!X)
    throw runtime_error("Failed to cast X to Tensor<T>.");
  if (!Y)
    throw runtime_error("Output tensor Y is not allocated.");
  Arithmetic_mml<T> arithmetic;

  if (approximate == "none") {
    arithmetic.elementwise(
        X,
        [](T x) {
          return static_cast<T>(0.5f * x * (1 + std::erf(x / std::sqrt(2))));
        },
        Y);
  } else {
    arithmetic.elementwise(
        X,
        [](T x) {
          return static_cast<T>(
              0.5 * x *
              (1 + std::tanh(std::sqrt(2 / M_PI) *
                             (x + 0.044715f * std::pow(x, 3)))));
        },
        Y);
  }
}
template <typename T> bool GeluNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void GeluNode<T>::setInputs(const array_mml<GeneralDataTypes> &inputs) {
  if (inputs.size() < 1)
    throw runtime_error("GeluNode expects at least one input: X.");
  auto valueX = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);
  if (!X || !valueX)
    throw runtime_error("Failed to cast X or input X to Tensor<T>.");
  X = *valueX;
}

template <typename T> bool GeluNode<T>::areOutputsFilled() const {
  if (!Y)
    return false;
  return Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> GeluNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{
      GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}