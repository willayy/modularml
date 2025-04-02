#pragma once
#include "datastructures/tensor_operations_module.hpp"

template <typename T>
void (*TensorOperationsModule::add_ptr)(
    const shared_ptr<const Tensor<T>> a, const shared_ptr<const Tensor<T>> b,
    shared_ptr<Tensor<T>> c) = mml_add;

template <typename T>
void (*TensorOperationsModule::subtract_ptr)(
    const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b,
    shared_ptr<Tensor<T>> c) = mml_subtract;

template <typename T>
void (*TensorOperationsModule::multiply_ptr)(
    const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c) =
    mml_multiply;

template <typename T>
bool (*TensorOperationsModule::equals_ptr)(
    const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b) =
    mml_equals;

template <typename T>
void (*TensorOperationsModule::elementwise_ptr)(
    const shared_ptr<const Tensor<T>> a, const function<T(T)> &f,
    const shared_ptr<Tensor<T>> c) = mml_elementwise;

template <typename T>
void (*TensorOperationsModule::elementwise_in_place_ptr)(
    const shared_ptr<Tensor<T>> a, const function<T(T)> &f) =
    mml_elementwise_in_place;

template <typename T>
void (*TensorOperationsModule::gemm_ptr)(
    int TA, int TB, int M, int N, int K, T ALPHA,
    shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B,
    int ldb, T BETA, shared_ptr<Tensor<T>> C, int ldc) = mml_gemm_inner_product;
  
template <typename T>
shared_ptr<Tensor<T>> (*TensorOperationsModule::gemm_onnx_ptr)(
    shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    optional<shared_ptr<Tensor<T>>> C) = mml_onnx_gemm_inner_product;

template <typename T>
int (*TensorOperationsModule::arg_max_ptr)(
    const shared_ptr<const Tensor<T>> a) = mml_arg_max;

template <typename T>
void TensorOperationsModule::set_add_ptr(
    void (*ptr)(const shared_ptr<const Tensor<T>> a,
                const shared_ptr<const Tensor<T>> b, shared_ptr<Tensor<T>> c)) {
  add_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_subtract_ptr(
    void (*ptr)(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b,
                shared_ptr<Tensor<T>> c)) {
  subtract_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_multiply_ptr(void (*ptr)(
    const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c)) {
  multiply_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_equals_ptr(
    bool (*ptr)(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b)) {
  equals_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_ptr(
    void (*ptr)(const shared_ptr<const Tensor<T>> a, const function<T(T)> &f,
                const shared_ptr<Tensor<T>> c)) {
  elementwise_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_in_place_ptr(
    void (*ptr)(const shared_ptr<Tensor<T>> a, const function<T(T)> &f)) {
  elementwise_in_place_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_ptr(
    void (*ptr)(int TA, int TB, int M, int N, int K, T ALPHA,
                shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B,
                int ldb, T BETA, shared_ptr<Tensor<T>> C, int ldc)) {
  gemm_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_onnx_ptr(shared_ptr<Tensor<T>> (*ptr)(
    shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha, float beta,
    int transA, int transB, optional<shared_ptr<Tensor<T>>> C)) {
  gemm_onnx_ptr<T> = ptr;
}

template <typename T>
void TensorOperationsModule::set_arg_max_ptr(
    int (*ptr)(const shared_ptr<const Tensor<T>> a)) {
  arg_max_ptr<T> = ptr;
}

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