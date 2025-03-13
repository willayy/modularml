#pragma once

#include "Add_node.hpp"

template <typename T>
AddNode<T>::AddNode(shared_ptr<const AbstractTensor> A, shared_ptr<const AbstractTensor> B,
                    shared_ptr<AbstractTensor> C)
    : A(A), B(B), C(C) {}

template <typename T>
void AddNode<T>::forward() {
  if (!areInputsFilled()) {
    throw runtime_error("AddNode forward called without inputs being set");
  }

  if (!A)
    throw runtime_error("Failed to cast A to Tensor<T>");
  if (!B)
    throw runtime_error("Failed to cast B to Tensor<T>");

    Arithmetic_mml<T> arithmetic;
  arithmetic.add(A, B, C);
}

template <typename T>
bool AddNode<T>::areInputsFilled() const {
  return A && A->get_size() > 0 && B && B->get_size() > 0;
}

template <typename T>
void AddNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() != 2) {
    throw runtime_error("AddNode expects 2 inputs");
  }

  auto valueA = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);
  auto valueB = std::get_if<shared_ptr<AbstractTensor>>(&inputs[1]);

  if (!valueA || !valueB) {
    throw runtime_error("Failed to cast inputs to the expected tensor types.");
  }

  A = *valueA;
  B = *valueB;
}

template <typename T>
bool AddNode<T>::areOutputsFilled() const {
  return C && C->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> AddNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(C))};
}