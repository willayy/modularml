#pragma once
#include "datastructures/tensor_operations_module.hpp"

// Initialize static std::function members with default implementations
template <typename T>
std::function<void(const shared_ptr<const Tensor<T>> a,
                   const shared_ptr<const Tensor<T>> b,
                   shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::add_ptr =
        mml_add<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<void(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b,
                   shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::subtract_ptr =
        mml_subtract<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<void(const shared_ptr<Tensor<T>> a, const T b,
                   shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::multiply_ptr =
        mml_multiply<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<bool(const shared_ptr<Tensor<T>> a,
                   const shared_ptr<Tensor<T>> b)>
    TensorOperationsModule::equals_ptr =
        mml_equals<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<void(const shared_ptr<const Tensor<T>> a, const function<T(T)> &f,
                   const shared_ptr<Tensor<T>> c)>
    TensorOperationsModule::elementwise_ptr =
        mml_elementwise<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<void(const shared_ptr<Tensor<T>> a, const function<T(T)> &f)>
    TensorOperationsModule::elementwise_in_place_ptr =
        mml_elementwise_in_place<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                   shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B,
                   int ldb, T BETA, shared_ptr<Tensor<T>> C, int ldc)>
    TensorOperationsModule::gemm_ptr =
        mml_gemm_inner_product<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<shared_ptr<Tensor<T>>(shared_ptr<Tensor<T>> A,
                                    shared_ptr<Tensor<T>> B, float alpha,
                                    float beta, int transA, int transB,
                                    optional<shared_ptr<Tensor<T>>> C)>
    TensorOperationsModule::gemm_onnx_ptr =
        mml_onnx_gemm_inner_product<T>; // NOSONAR - Not a global variable

template <typename T>
std::function<int(const shared_ptr<const Tensor<T>> a)>
    TensorOperationsModule::arg_max_ptr =
        mml_arg_max<T>; // NOSONAR - Not a global variable

// Setter implementations
template <typename T>
void TensorOperationsModule::set_add_ptr(
    std::function<void(const shared_ptr<const Tensor<T>> a,
                       const shared_ptr<const Tensor<T>> b,
                       shared_ptr<Tensor<T>> c)>
        ptr) {
  add_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_subtract_ptr(
    std::function<void(const shared_ptr<Tensor<T>> a,
                       const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c)>
        ptr) {
  subtract_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_multiply_ptr(
    std::function<void(const shared_ptr<Tensor<T>> a, const T b,
                       shared_ptr<Tensor<T>> c)>
        ptr) {
  multiply_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_equals_ptr(
    std::function<bool(const shared_ptr<Tensor<T>> a,
                       const shared_ptr<Tensor<T>> b)>
        ptr) {
  equals_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_ptr(
    std::function<void(const shared_ptr<const Tensor<T>> a,
                       const function<T(T)> &f, const shared_ptr<Tensor<T>> c)>
        ptr) {
  elementwise_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_in_place_ptr(
    std::function<void(const shared_ptr<Tensor<T>> a, const function<T(T)> &f)>
        ptr) {
  elementwise_in_place_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_ptr(
    std::function<void(int TA, int TB, int M, int N, int K, T ALPHA,
                       shared_ptr<Tensor<T>> A, int lda,
                       shared_ptr<Tensor<T>> B, int ldb, T BETA,
                       shared_ptr<Tensor<T>> C, int ldc)>
        ptr) {
  gemm_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_onnx_ptr(
    std::function<shared_ptr<Tensor<T>>(
        shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha,
        float beta, int transA, int transB, optional<shared_ptr<Tensor<T>>> C)>
        ptr) {
  gemm_onnx_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_arg_max_ptr(
    std::function<int(const shared_ptr<const Tensor<T>> a)> ptr) {
  arg_max_ptr<T> = ptr;
}

// Function implementations
template <typename T>
void TensorOperationsModule::add(const shared_ptr<const Tensor<T>> a,
                                 const shared_ptr<const Tensor<T>> b,
                                 shared_ptr<Tensor<T>> c) {
  add_ptr<T>(a, b, c);
}

template <typename T>
void TensorOperationsModule::subtract(const shared_ptr<Tensor<T>> a,
                                      const shared_ptr<Tensor<T>> b,
                                      shared_ptr<Tensor<T>> c) {
  subtract_ptr<T>(a, b, c);
}

template <typename T>
void TensorOperationsModule::multiply(const shared_ptr<Tensor<T>> a, const T b,
                                      shared_ptr<Tensor<T>> c) {
  multiply_ptr<T>(a, b, c);
}

template <typename T>
bool TensorOperationsModule::equals(const shared_ptr<Tensor<T>> a,
                                    const shared_ptr<Tensor<T>> b) {
  return equals_ptr<T>(a, b);
}

template <typename T>
void TensorOperationsModule::elementwise(const shared_ptr<const Tensor<T>> a,
                                         function<T(T)> f,
                                         const shared_ptr<Tensor<T>> c) {
  elementwise_ptr<T>(a, f, c);
}

template <typename T>
void TensorOperationsModule::elementwise_in_place(const shared_ptr<Tensor<T>> a,
                                                  function<T(T)> f) {
  elementwise_in_place_ptr<T>(a, f);
}

template <typename T>
void TensorOperationsModule::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                                  shared_ptr<Tensor<T>> A, int lda,
                                  shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                  shared_ptr<Tensor<T>> C, int ldc) {
  gemm_ptr<T>(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

template <typename T>
shared_ptr<Tensor<T>> TensorOperationsModule::gemm_onnx(
    shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha, float beta,
    int transA, int transB, optional<shared_ptr<Tensor<T>>> C) {
  return gemm_onnx_ptr<T>(A, B, alpha, beta, transA, transB, C);
}

template <typename T>
int TensorOperationsModule::arg_max(const shared_ptr<const Tensor<T>> a) {
  return arg_max_ptr<T>(a);
}