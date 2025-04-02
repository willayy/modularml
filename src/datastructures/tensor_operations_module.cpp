#pragma once

#include "datastructures/tensor_operations_module.hpp"

template <typename T>
void TensorOperationsModule::set_add_ptr(
    void (*ptr)(const shared_ptr<const Tensor<T>> a,
                const shared_ptr<const Tensor<T>> b, shared_ptr<Tensor<T>> c)) {
  this->add_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_subtract_ptr(
    void (*ptr)(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b,
                shared_ptr<Tensor<T>> c)) {
  this->subtract_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_multiply_ptr(void (*ptr)(
    const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c)) {
  this->multiply_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_equals_ptr(
    bool (*ptr)(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b)) {
  this->equals_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_ptr(
    void (*ptr)(const shared_ptr<const Tensor<T>> a, const function<T(T)> &f,
                const shared_ptr<Tensor<T>> c)) {
  this->elementwise_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_elementwise_in_place_ptr(
    void (*ptr)(const shared_ptr<Tensor<T>> a, const function<T(T)> &f)) {
  this->elementwise_in_place_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_ptr(
    void (*ptr)(int TA, int TB, int M, int N, int K, T ALPHA,
                shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B,
                int ldb, T BETA, shared_ptr<Tensor<T>> C, int ldc)) {
  this->gemm_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_gemm_onnx_ptr(shared_ptr<Tensor<T>> (*ptr)(
    shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha, float beta,
    int transA, int transB, optional<shared_ptr<Tensor<T>>> C)) {
  this->gemm_onnx_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::set_arg_max_ptr(
    int (*ptr)(const shared_ptr<const Tensor<T>> a)) {
  this->arg_max_ptr = ptr;
}

template <typename T>
void TensorOperationsModule::add(const shared_ptr<const Tensor<T>> a,
                                 const shared_ptr<const Tensor<T>> b,
                                 shared_ptr<Tensor<T>> c) {
  this->add_ptr(a, b, c);
}

template <typename T>
void TensorOperationsModule::subtract(const shared_ptr<Tensor<T>> a,
                                      const shared_ptr<Tensor<T>> b,
                                      shared_ptr<Tensor<T>> c) {
  this->subtract_ptr(a, b, c);
}

template <typename T>
void TensorOperationsModule::multiply(const shared_ptr<Tensor<T>> a, const T b,
                                      shared_ptr<Tensor<T>> c) {
  this->multiply_ptr(a, b, c);
}

template <typename T>
bool TensorOperationsModule::equals(const shared_ptr<Tensor<T>> a,
                                    const shared_ptr<Tensor<T>> b) {
  return this->equals_ptr(a, b);
}

template <typename T>
void TensorOperationsModule::elementwise(const shared_ptr<const Tensor<T>> a,
                                         function<T(T)> f,
                                         const shared_ptr<Tensor<T>> c) {
  this->elementwise_ptr(a, f, c);
}

template <typename T>
void TensorOperationsModule::elementwise_in_place(const shared_ptr<Tensor<T>> a,
                                                  function<T(T)> f) {
  this->elementwise_in_place_ptr(a, f);
}

template <typename T>
void TensorOperationsModule::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                                  shared_ptr<Tensor<T>> A, int lda,
                                  shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                  shared_ptr<Tensor<T>> C, int ldc) {
  this->gemm_ptr(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

template <typename T>
shared_ptr<Tensor<T>> TensorOperationsModule::gemm_onnx(
    shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha, float beta,
    int transA, int transB, optional<shared_ptr<Tensor<T>>> C) {
  return this->gemm_onnx_ptr(A, B, alpha, beta, transA, transB, C);
}

template <typename T>
int TensorOperationsModule::arg_max(const shared_ptr<const Tensor<T>> a) {
  return this->arg_max_ptr(a);
}