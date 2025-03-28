#pragma once

#include "backend/mml_onnx_gemm.hpp"

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_inner_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                          float alpha, float beta, int transA, int transB,
                                                          optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_inner_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_outer_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                          float alpha, float beta, int transA, int transB,
                                                          optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_outer_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_row_wise_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                             float alpha, float beta, int transA, int transB,
                                                             optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_row_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_col_wise_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                             float alpha, float beta, int transA, int transB,
                                                             optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_col_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_blocked(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                    float alpha, float beta, int transA, int transB,
                                                    optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_blocked(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_avx(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                float alpha, float beta, int transA, int transB,
                                                optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_avx(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_avx512(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                   float alpha, float beta, int transA, int transB,
                                                   optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_avx512(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_intel_MKL(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                      float alpha, float beta, int transA, int transB,
                                                      optional<shared_ptr<Tensor<T>>> C) {
  unique_ptr<GemmModule<T>> gm = make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const uli M = shape_A[0];
  const uli N = shape_B[1];
  const uli K = shape_A[1];
  const uli lda = K;
  const uli ldb = N;
  const uli ldc = N;
  shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_intel_MKL(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}
