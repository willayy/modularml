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


  auto A_shape = A->get_shape();
  auto B_shape = B->get_shape();
  auto A_rank = A_shape.size();
  auto B_rank = B_shape.size();
  auto max_rank = std::max(A_rank, B_rank);
  bool broadcast_comp = true;

  // Check if broadcasting is possible
  for (int i = 0; i < max_rank; i++) {
    int dim_A = (i < A_rank) ? A_shape[A_rank - 1 - i] : 1;
    int dim_B = (i < B_rank) ? B_shape[B_rank - 1 - i] : 1;

    // Valid if dimensions match or one of them is 1
    if (dim_A != dim_B && dim_A != 1 && dim_B != 1) {
      broadcast_comp = false;  // Incompatible for broadcasting
    }
  }

  Arithmetic_mml<T> arithmetic;

  // Valid case:
  if (A_shape == B_shape) {
    arithmetic.add(A, B, C);
    // Broadcasting case:
  } else if (broadcast_comp) {
    throw runtime_error("BROADCASTING NOT IMPLEMENTED YET");
    // Invalid case:
  } else {
    throw runtime_error("Incompatible shapes for addition attempt in AddNode. Broadcasting impossible.");
  }
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