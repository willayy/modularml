#pragma once

#include "backend/mml_onnx_gemm.hpp"

template <typename T>
std::shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_inner_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_inner_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta,
                         C_p, ldc);
  return C_p;
}

template <typename T>
std::shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_outer_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_outer_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta,
                         C_p, ldc);
  return C_p;
}

template <typename T>
std::shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_row_wise_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_row_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
std::shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_col_wise_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_col_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
std::shared_ptr<Tensor<T>>
OnnxGemm_mml<T>::gemm_blocked(std::shared_ptr<Tensor<T>> A,
                              std::shared_ptr<Tensor<T>> B, float alpha,
                              float beta, int transA, int transB,
                              std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_blocked(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                   ldc);
  return C_p;
}

template <typename T>
std::shared_ptr<Tensor<T>>
OnnxGemm_mml<T>::gemm_avx(std::shared_ptr<Tensor<T>> A,
                          std::shared_ptr<Tensor<T>> B, float alpha, float beta,
                          int transA, int transB,
                          std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_avx(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
std::shared_ptr<Tensor<T>>
OnnxGemm_mml<T>::gemm_avx512(std::shared_ptr<Tensor<T>> A,
                             std::shared_ptr<Tensor<T>> B, float alpha,
                             float beta, int transA, int transB,
                             std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_avx512(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                  ldc);
  return C_p;
}

template <typename T>
std::shared_ptr<Tensor<T>>
OnnxGemm_mml<T>::gemm_intel_MKL(std::shared_ptr<Tensor<T>> A,
                                std::shared_ptr<Tensor<T>> B, float alpha,
                                float beta, int transA, int transB,
                                std::optional<std::shared_ptr<Tensor<T>>> C) {
  std::unique_ptr<GemmModule<T>> gm = std::make_unique<Gemm_mml<T>>();
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const size_t M = shape_A[0];
  const size_t N = shape_B[1];
  const size_t K = shape_A[1];
  const size_t lda = K;
  const size_t ldb = N;
  const size_t ldc = N;
  std::shared_ptr<Tensor<T>> C_p = C.has_value() ? *C : tensor_mml_p<T>({M, N});
  gm->gemm_intel_MKL(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                     ldc);
  return C_p;
}
