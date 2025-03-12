#pragma once

#include "gemm_node.hpp"

template <typename T>
GemmNode<T>::GemmNode(shared_ptr<AbstractTensor> A,
                      shared_ptr<AbstractTensor> B,
                      shared_ptr<AbstractTensor> Y,
                      optional<shared_ptr<AbstractTensor>> C,
                      float alpha, float beta,
                      int transA, int transB)
    : A(A), B(B), C(C), Y(Y), alpha(alpha), beta(beta), transA(transA), transB(transB) {}

template <typename T>
void GemmNode<T>::forward() {
  if (!areInputsFilled())
    throw runtime_error("GemmNode inputs are not fully set.");

  auto shapeA = A->get_shape();
  if (shapeA.size() < 2)
    throw runtime_error("Tensor A must be at least 2D.");

  int M = shapeA[0];  // Number of rows.
  int K = shapeA[1];  // Number of columns of A.

  auto shapeB = B->get_shape();
  if (shapeB.size() < 2)
    throw runtime_error("Tensor B must be at least 2D.");
  if (shapeB[0] != K)
    throw runtime_error("GemmNode: Dimension mismatch between A and B.");

  int N = shapeB[1];  // Number of columns of B.

  int lda = K;
  int ldb = N;
  int ldc = N;

  // Handling optional C tensor not implemented directly in gemm_inner_product.
  // Will have to be done here instead by constructing suboptimal concrete tensor.
  // Gemm_inner_product could be modified to handle optional C tensor and take output Y.
  shared_ptr<Tensor_mml<T>> C_ptr;
  if (C.has_value() && C.value()) {
    C_ptr = std::dynamic_pointer_cast<Tensor_mml<T>>(C.value());
    if (!C_ptr)
      throw runtime_error("GemmNode: Failed to cast optional C to Tensor_mml<T>.");
  } else {
    Tensor_mml<T> zero_tensor({M, N});
    zero_tensor.fill(static_cast<T>(0));
    C_ptr = make_shared<Tensor_mml<T>>(zero_tensor);
  }

  Gemm_mml<T> gemm;
  gemm.gemm_inner_product(0, 0, M, N, K, static_cast<T>(alpha),
                          A, lda,
                          B, ldb,
                          static_cast<T>(beta),
                          C_ptr, ldc);

  *Y = *C_ptr;
}

template <typename T>
bool GemmNode<T>::areInputsFilled() const {
  return A && A->get_size() > 0 &&
         B && B->get_size() > 0 &&
         (!C.has_value() || (C.value() && C.value()->get_size() > 0));
}

template <typename T>
void GemmNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() > 0) {
    auto valueA = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
    *A = *valueA;
  }

  if (inputs.size() > 1) {
    auto valueB = std::get<std::shared_ptr<AbstractTensor>>(inputs[1]);
    *B = *valueB;
  }

  if (inputs.size() > 2 && C.has_value()) {
    auto valueC = std::get<std::shared_ptr<AbstractTensor>>(inputs[2]);
    *C.value() = *valueC;
  }
}

template <typename T>
bool GemmNode<T>::areOutputsFilled() const {
  return Y && Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> GemmNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}