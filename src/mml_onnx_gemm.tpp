#include "include/mml_onnx_gemm.hpp"

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_inner_product(float alpha, float beta,
                                                          int transA, int transB,
                                                          shared_ptr<Tensor<T>> A,
                                                          shared_ptr<Tensor<T>> B,
                                                          optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_outer_product(float alpha, float beta,
                                                          int transA, int transB,
                                                          shared_ptr<Tensor<T>> A,
                                                          shared_ptr<Tensor<T>> B,
                                                          optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_row_wise_product(float alpha, float beta,
                                                             int transA, int transB,
                                                             shared_ptr<Tensor<T>> A,
                                                             shared_ptr<Tensor<T>> B,
                                                             optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_col_wise_product(float alpha, float beta,
                                                             int transA, int transB,
                                                             shared_ptr<Tensor<T>> A,
                                                             shared_ptr<Tensor<T>> B,
                                                             optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_blocked(float alpha, float beta,
                                                    int transA, int transB,
                                                    shared_ptr<Tensor<T>> A,
                                                    shared_ptr<Tensor<T>> B,
                                                    optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_avx(float alpha, float beta,
                                                int transA, int transB,
                                                shared_ptr<Tensor<T>> A,
                                                shared_ptr<Tensor<T>> B,
                                                optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_avx512(float alpha, float beta,
                                                   int transA, int transB,
                                                   shared_ptr<Tensor<T>> A,
                                                   shared_ptr<Tensor<T>> B,
                                                   optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}

template <typename T>
shared_ptr<Tensor<T>> OnnxGemm_mml<T>::gemm_intel_MKL(float alpha, float beta,
                                                      int transA, int transB,
                                                      shared_ptr<Tensor<T>> A,
                                                      shared_ptr<Tensor<T>> B,
                                                      optional<shared_ptr<Tensor<T>>> C) {
  // Placeholder implementation
  return nullptr;
}
