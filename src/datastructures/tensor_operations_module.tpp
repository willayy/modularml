#pragma once
#include "datastructures/tensor_operations_module.hpp"

#if defined(USE_BLOCKED_GEMM)  // Use blocked gemm routine
template <typename T>
std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                   shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B,
                   int ldb, T BETA, shared_ptr<Tensor<T>> C, int ldc)>
    TensorOperations::gemm_ptr =
        mml_gemm_blocked<T>;
#endif

// Setter implementations
template <typename... Ts>
void TensorOperations::set_add_ptr(toft::add_func<Ts>... ptr) {
  (..., (add_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_subtract_ptr(toft::subtract_func<Ts>... ptr) {
  (..., (subtract_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_multiply_ptr(toft::multiply_func<Ts>... ptr) {
  (..., (multiply_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_equals_ptr(toft::equals_func<Ts>... ptr) {
  (..., (equals_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_elementwise_ptr(toft::elementwise_func<Ts>... ptr) {
  (..., (elementwise_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_elementwise_in_place_ptr(
    toft::elementwise_in_place_func<Ts>... ptr) {
  (..., (elementwise_in_place_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_gemm_ptr(toft::gemm_func<Ts>... ptr) {
  (..., (gemm_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_gemm_onnx_ptr(toft::gemm_onnx_func<Ts>... ptr) {
  (..., (gemm_onnx_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_arg_max_ptr(toft::arg_max_func<Ts>... ptr) {
  (..., (arg_max_ptr<Ts> = ptr));
}

template <typename... Ts>
void TensorOperations::set_sliding_window_ptr(
    toft::sliding_window_func<Ts>... ptr) {
  (..., (sliding_window_ptr<Ts> = ptr));
}

// Function implementations
template <typename T>
void TensorOperations::add(const std::shared_ptr<const Tensor<T>> a,
                           const std::shared_ptr<const Tensor<T>> b,
                           std::shared_ptr<Tensor<T>> c) {
  add_ptr<T>(a, b, c);
}

template <typename T>
void TensorOperations::subtract(const std::shared_ptr<Tensor<T>> a,
                                const std::shared_ptr<Tensor<T>> b,
                                std::shared_ptr<Tensor<T>> c) {
  subtract_ptr<T>(a, b, c);
}

template <typename T>
void TensorOperations::multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                                std::shared_ptr<Tensor<T>> c) {
  multiply_ptr<T>(a, b, c);
}

template <typename T>
bool TensorOperations::equals(const std::shared_ptr<Tensor<T>> a,
                              const std::shared_ptr<Tensor<T>> b) {
  return equals_ptr<T>(a, b);
}

template <typename T>
void TensorOperations::elementwise(const std::shared_ptr<const Tensor<T>> a,
                                   std::function<T(T)> f,
                                   const std::shared_ptr<Tensor<T>> c) {
  elementwise_ptr<T>(a, f, c);
}

template <typename T>
void TensorOperations::elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                            std::function<T(T)> f) {
  elementwise_in_place_ptr<T>(a, f);
}

template <typename T>
void TensorOperations::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                            std::shared_ptr<Tensor<T>> A, int lda,
                            std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            std::shared_ptr<Tensor<T>> C, int ldc) {
  gemm_ptr<T>(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorOperations::gemm_onnx(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  return gemm_onnx_ptr<T>(A, B, alpha, beta, transA, transB, C);
}

template <typename T>
int TensorOperations::arg_max(const std::shared_ptr<const Tensor<T>> a) {
  return arg_max_ptr<T>(a);
}

template <typename T>
void TensorOperations::sliding_window(
    const array_mml<size_t>& in_shape, const array_mml<size_t>& out_shape,
    const std::vector<int>& kernel_shape, const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const std::vector<std::pair<int, int>>& pads,
    const std::function<void(const std::vector<std::vector<size_t>>&,
                             const std::vector<size_t>&)>& window_f) {
  return sliding_window_ptr<T>(in_shape, out_shape, kernel_shape, strides,
                               dilations, pads, window_f);
}