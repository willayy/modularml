#pragma once
#include "datastructures/tensor_operations_module.hpp"

// Initialize static std::function members with default implementations
template <typename T>
std::function<void(const std::shared_ptr<const Tensor<T>> a,
                   const std::shared_ptr<const Tensor<T>> b,
                   std::shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::add_ptr =
        mml_add<T>;  // NOSONAR - Not a global variable

template <typename T>
std::function<void(const std::shared_ptr<Tensor<T>> a,
                   const std::shared_ptr<Tensor<T>> b,
                   std::shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::subtract_ptr =
        mml_subtract<T>;  // NOSONAR - Not a global variable

template <typename T>
std::function<void(const std::shared_ptr<Tensor<T>> a, const T b,
                   std::shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::multiply_ptr =
        mml_multiply<T>;  // NOSONAR - Not a global variable

template <typename T>
std::function<bool(const std::shared_ptr<Tensor<T>> a,
                   const std::shared_ptr<Tensor<T>> b)>
    TensorOperationsModule::equals_ptr =
        mml_equals<T>;  // NOSONAR - Not a global variable

template <typename T>
std::function<void(const std::shared_ptr<const Tensor<T>> a,
                   const std::function<T(T)> &f,
                   const std::shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::elementwise_ptr =
        mml_elementwise<T>;  // NOSONAR - Not a global variable

template <typename T>
std::function<void(const std::shared_ptr<Tensor<T>> a,
                   const std::function<T(T)> &f)>
    TensorOperationsModule::elementwise_in_place_ptr =
        mml_elementwise_in_place<T>;  // NOSONAR - Not a global variable

#if defined(USE_BLOCKED_GEMM)  // Use blocked gemm routine
template <typename T>
std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                   std::shared_ptr<Tensor<T>> A, int lda,
                   std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                   std::shared_ptr<Tensor<T>> C, int ldc)>
    TensorOperationsModule::gemm_ptr =
        mml_gemm_blocked<T>;  // NOSONAR - Not a global variable
#elif defined(USE_AVX_GEMM)
template <typename T>
std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                   std::shared_ptr<Tensor<T>> A, int lda, std::shared_ptr<Tensor<T>> B,
                   int ldb, T BETA, std::shared_ptr<Tensor<T>> C, int ldc)>
    TensorOperationsModule::gemm_ptr =
        mml_gemm_avx<T>;  // NOSONAR - Not a global variable
#elif defined(USE_AVX512_GEMM)
template <typename T>
std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                   std::shared_ptr<Tensor<T>> A, int lda, std::shared_ptr<Tensor<T>> B,
                   int ldb, T BETA, std::shared_ptr<Tensor<T>> C, int ldc)>
    TensorOperationsModule::gemm_ptr =
        mml_gemm_avx512<T>;  // NOSONAR - Not a global variable
#else  // Default naive gemm implementation
template <typename T>
std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                   std::shared_ptr<Tensor<T>> A, int lda, std::shared_ptr<Tensor<T>> B,
                   int ldb, T BETA, std::shared_ptr<Tensor<T>> C, int ldc)>
    TensorOperationsModule::gemm_ptr =
        mml_gemm_inner_product<T>;  // NOSONAR - Not a global variable
#endif

template <typename T>
std::function<std::shared_ptr<Tensor<T>>(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C)>
    TensorOperationsModule::gemm_onnx_ptr =
        mml_onnx_gemm_inner_product<T>;  // NOSONAR - Not a global variable

template <typename T>
std::function<int(const std::shared_ptr<const Tensor<T>> a)>
    TensorOperationsModule::arg_max_ptr =
        mml_arg_max<T>;  // NOSONAR - Not a global variable

template <typename T>
std::function<void(const std::shared_ptr<const Tensor<T>>& input,
                   std::shared_ptr<Tensor<T>>& output,
                   const std::optional<std::shared_ptr<Tensor<int64_t>>>& indices_out,
                   const std::vector<int>& kernel_shape,
                   const std::vector<int>& strides,
                   const std::vector<int>& dilations,
                   const std::vector<std::pair<int, int>>& pads,
                   WindowOpFn<T> window_fn)>
    TensorOperationsModule::sliding_window_ptr =
        mml_sliding_window<T>; // NOSONAR - Not a global variable

// Setter implementations
template <typename T>
void TensorOperationsModule::set_add_ptr(
    std::function<void(const std::shared_ptr<const Tensor<T>> a,
                       const std::shared_ptr<const Tensor<T>> b,
                       std::shared_ptr<Tensor<T>> c)>
        ptr) {
    add_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_subtract_ptr(
    std::function<void(const std::shared_ptr<Tensor<T>> a,
                       const std::shared_ptr<Tensor<T>> b,
                       std::shared_ptr<Tensor<T>> c)>
        ptr) {
    subtract_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_multiply_ptr(
    std::function<void(const std::shared_ptr<Tensor<T>> a, const T b,
                       std::shared_ptr<Tensor<T>> c)>
        ptr) {
    multiply_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_equals_ptr(
    std::function<bool(const std::shared_ptr<Tensor<T>> a,
                       const std::shared_ptr<Tensor<T>> b)>
        ptr) {
    equals_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_ptr(
    std::function<void(const std::shared_ptr<const Tensor<T>> a,
                       const std::function<T(T)> &f,
                       const std::shared_ptr<Tensor<T>> c)>
        ptr) {
    elementwise_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_in_place_ptr(
    std::function<void(const std::shared_ptr<Tensor<T>> a,
                       const std::function<T(T)> &f)>
        ptr) {
    elementwise_in_place_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_ptr(
    std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                       std::shared_ptr<Tensor<T>> A, int lda,
                       std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                       std::shared_ptr<Tensor<T>> C, int ldc)>
        ptr) {
    gemm_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_onnx_ptr(
    std::function<std::shared_ptr<Tensor<T>>(
        std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
        float beta, int transA, int transB,
        std::optional<std::shared_ptr<Tensor<T>>> C)>
        ptr) {
    gemm_onnx_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_arg_max_ptr(
    std::function<int(const std::shared_ptr<const Tensor<T>> a)> ptr) {
  arg_max_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_sliding_window_ptr(
    std::function<void(const std::shared_ptr<const Tensor<T>>& input,
                       std::shared_ptr<Tensor<T>>& output,
                       const std::optional<std::shared_ptr<Tensor<int64_t>>>& indices_out,
                       const std::vector<int>& kernel_shape,
                       const std::vector<int>& strides,
                       const std::vector<int>& dilations,
                       const std::vector<std::pair<int, int>>& pads,
                       WindowOpFn<T> window_fn)>
        ptr) {
  sliding_window_ptr<T> = ptr;
}

// Function implementations
template <typename T>
void TensorOperationsModule::add(const std::shared_ptr<const Tensor<T>> a,
                                 const std::shared_ptr<const Tensor<T>> b,
                                 std::shared_ptr<Tensor<T>> c) {
  add_ptr<T>(a, b, c);
}

template <typename T>
void TensorOperationsModule::subtract(const std::shared_ptr<Tensor<T>> a,
                                      const std::shared_ptr<Tensor<T>> b,
                                      std::shared_ptr<Tensor<T>> c) {
  subtract_ptr<T>(a, b, c);
}

template <typename T>
void TensorOperationsModule::multiply(const std::shared_ptr<Tensor<T>> a,
                                      const T b, std::shared_ptr<Tensor<T>> c) {
  multiply_ptr<T>(a, b, c);
}

template <typename T>
bool TensorOperationsModule::equals(const std::shared_ptr<Tensor<T>> a,
                                    const std::shared_ptr<Tensor<T>> b) {
  return equals_ptr<T>(a, b);
}

template <typename T>
void TensorOperationsModule::elementwise(
    const std::shared_ptr<const Tensor<T>> a, std::function<T(T)> f,
    const std::shared_ptr<Tensor<T>> c) {
  elementwise_ptr<T>(a, f, c);
}

template <typename T>
void TensorOperationsModule::elementwise_in_place(
    const std::shared_ptr<Tensor<T>> a, std::function<T(T)> f) {
  elementwise_in_place_ptr<T>(a, f);
}

template <typename T>
void TensorOperationsModule::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                                  std::shared_ptr<Tensor<T>> A, int lda,
                                  std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                  std::shared_ptr<Tensor<T>> C, int ldc) {
  gemm_ptr<T>(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorOperationsModule::gemm_onnx(std::shared_ptr<Tensor<T>> A,
                                  std::shared_ptr<Tensor<T>> B, float alpha,
                                  float beta, int transA, int transB,
                                  std::optional<std::shared_ptr<Tensor<T>>> C) {
  return gemm_onnx_ptr<T>(A, B, alpha, beta, transA, transB, C);
}

template <typename T>
int TensorOperationsModule::arg_max(const std::shared_ptr<const Tensor<T>> a) {
  return arg_max_ptr<T>(a);
}

template <typename T>
void TensorOperationsModule::sliding_window(const std::shared_ptr<const Tensor<T>>& input,
                                            std::shared_ptr<Tensor<T>>& output,
                                            const std::optional<std::shared_ptr<Tensor<int64_t>>>& indices_out,
                                            const std::vector<int>& kernel_shape,
                                            const std::vector<int>& strides,
                                            const std::vector<int>& dilations,
                                            const std::vector<std::pair<int, int>>& pads,
                                            WindowOpFn<T> window_fn) {
  return sliding_window_ptr<T>(input, output, indices_out, kernel_shape, strides, dilations, pads, window_fn);
}