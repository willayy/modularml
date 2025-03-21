#pragma once

#include "ELU_node.hpp"

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
  Arithmetic_mml<T> arithmetic;
  arithmetic.elementwise(
      X, [](T x) { return x < 0 ? alpha * (exp(x) - 1) : x; }, Y);
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