#pragma once

#include "Add_node.hpp"

template <typename T>
AddNode<T>::AddNode(shared_ptr<const AbstractTensor> A,
                    shared_ptr<const AbstractTensor> B,
                    shared_ptr<AbstractTensor> C)
    : A(A), B(B), C(C) {}

template <typename T> void AddNode<T>::forward() {
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
  for (uli i = 0; i < max_rank; i++) {
    uli dim_A = (i < A_rank) ? A_shape[A_rank - 1 - i] : 1;
    uli dim_B = (i < B_rank) ? B_shape[B_rank - 1 - i] : 1;

    // Valid if dimensions match or one of them is 1
    if (dim_A != dim_B && dim_A != 1 && dim_B != 1) {
      broadcast_comp = false; // Incompatible for broadcasting
    }
  }

  Arithmetic_mml<T> arithmetic;

  // Valid case:
  if (A_shape == B_shape) {
    if (C->get_shape() != A_shape) {
      C->reshape(
          A_shape); // Reshape output tensor to be the same as input tensors
    }
    arithmetic.add(A, B, C);
    // Broadcasting case:
  } else if (broadcast_comp) {
    broadcast_addition();
    // Invalid case:
  } else {
    throw runtime_error("Incompatible shapes for addition attempt in AddNode. "
                        "Broadcasting impossible.");
  }
}

template <typename T> bool AddNode<T>::areInputsFilled() const {
  return A && A->get_size() > 0 && B && B->get_size() > 0;
}

template <typename T>
void AddNode<T>::setInputs(const array_mml<GeneralDataTypes> &inputs) {
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

template <typename T> bool AddNode<T>::areOutputsFilled() const {
  return C && C->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> AddNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{
      GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(C))};
}

template <typename T> void AddNode<T>::broadcast_addition() const {
  auto A_shape = A->get_shape();
  auto B_shape = B->get_shape();
  auto A_rank = A_shape.size();
  auto B_rank = B_shape.size();
  auto max_rank = std::max(A_rank, B_rank);

  // Compute output shape based on broadcasting rules
  array_mml<uli> output_shape(max_rank);
  std::fill(output_shape.begin(), output_shape.end(), 1);
  for (uli i = 0; i < max_rank; i++) {
    uli dim_A = (i < A_rank) ? A_shape[A_rank - 1 - i] : 1;
    uli dim_B = (i < B_rank) ? B_shape[B_rank - 1 - i] : 1;

    switch ((dim_A == dim_B) ? 0 : (dim_A == 1) ? 1 : (dim_B == 1) ? 2 : 3) {
    case 0:
      output_shape[max_rank - 1 - i] = dim_A;
      break;
    case 1:
      output_shape[max_rank - 1 - i] = dim_B;
      break;
    case 2:
      output_shape[max_rank - 1 - i] = dim_A;
      break;
    default:
      throw std::runtime_error("Incompatible shapes for broadcasting.");
    }
  }

  C->reshape(output_shape);

  vector<uli> A_strides(A_rank, 1);
  vector<uli> B_strides(B_rank, 1);
  vector<uli> output_strides(max_rank, 1);

  // Compute strides for each tensor
  for (uli i = max_rank - 2; ((int) i) >= 0; --i) {
    output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
  }
  for (uli i = A_rank - 2; ((int) i) >= 0; --i) {
    A_strides[i] = A_strides[i + 1] * A_shape[i + 1];
  }
  for (uli i = B_rank - 2; ((int) i) >= 0; --i) {
    B_strides[i] = B_strides[i + 1] * B_shape[i + 1];
  }

  // Iterate through the output tensor
  for (uli flat_idx = 0; flat_idx < C->get_size(); flat_idx++) {
    uli A_idx = 0, B_idx = 0;
    uli remaining = flat_idx;

    // Compute multi-dimensional indices on the fly
    for (uli j = 0; j < max_rank; j++) {
      uli coord = remaining / output_strides[j]; // Extract coordinate for dim j
      remaining %= output_strides[j];

      uli dim_A = (j < A_rank) ? A_shape[A_rank - max_rank + j] : 1;
      uli dim_B = (j < B_rank) ? B_shape[B_rank - max_rank + j] : 1;

      if (dim_A > 1)
        A_idx += coord * A_strides[j];
      if (dim_B > 1)
        B_idx += coord * B_strides[j];
    }

    // Perform element-wise addition
    T value_A = (*A)[A_idx];
    T value_B = (*B)[B_idx];
    (*C)[flat_idx] = value_A + value_B;
  }
}