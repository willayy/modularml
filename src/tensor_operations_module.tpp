#pragma once

#include "tensor_operations_module.hpp"

template <typename T>
TensorOperationsModule<T> &TensorOperationsModule<T>::getInstance() {
  static TensorOperationsModule instance;
  return instance;
}

template <typename T>
void TensorOperationsModule<T>::set_operation_func(string id,
                                                   function<void()> func) {
  if (id == "gemm") {
    static_assert(
        std::is_same_v<decltype(func), void (*)(int, int, int, int, int, T,
                                                shared_ptr<Tensor<T>>, int,
                                                shared_ptr<Tensor<T>>, int, T,
                                                shared_ptr<Tensor<T>>, int)>,
        "Function signature does not match gemm.");
    gemm_ptr = func
  } else if (id == "gemm_onnx") {
    static_assert(
        std::is_same_v<decltype(func),
                       shared_ptr<Tensor<T>> (*)(
                           shared_ptr<Tensor<T>>, shared_ptr<Tensor<T>>, float,
                           float, int, int, optional<shared_ptr<Tensor<T>>>)>,
        "Function signature does not match gemm_onnx.");
    gemm_onnx_ptr = func;
  } else if (id == "add") {
    static_assert(std::is_same_v<decltype(func),
                                 void (*)(const shared_ptr<const Tensor<T>>,
                                          const shared_ptr<const Tensor<T>>,
                                          shared_ptr<Tensor<T>>) const>,
                  "Function signature does not match add.");
    add_ptr = func;
  } else if (id == "subtract") {
    static_assert(
        std::is_same_v<decltype(func), void (*)(const shared_ptr<Tensor<T>>,
                                                const shared_ptr<Tensor<T>>,
                                                shared_ptr<Tensor<T>>) const>,
        "Function signature does not match subtract.");
    subtract_ptr = func;
  } else if (id == "multiply") {
    static_assert(
        std::is_same_v<decltype(func), void (*)(const shared_ptr<Tensor<T>>, T,
                                                shared_ptr<Tensor<T>>) const>,
        "Function signature does not match multiply.");
    multiply_ptr = func;
  } else if (id == "equals") {
    static_assert(std::is_same_v<decltype(func),
                                 bool (*)(const shared_ptr<Tensor<T>>,
                                          const shared_ptr<Tensor<T>>) const>,
                  "Function signature does not match equals.");
    equals_ptr = func;
  } else if (id == "elementwise") {
    static_assert(
        std::is_same_v<decltype(func),
                       void (*)(const shared_ptr<const Tensor<T>>, T (*)(T),
                                const shared_ptr<Tensor<T>>) const>,
        "Function signature does not match elementwise.");
    elementwise_ptr = func;
  } else if (id == "elementwise_in_place") {
    static_assert(
        std::is_same_v<decltype(func),
                       void (*)(const shared_ptr<Tensor<T>>, T (*)(T)) const>,
        "Function signature does not match elementwise_in_place.");
    elementwise_in_place_ptr = func;
  } else {
    throw invalid_argument("Invalid function id.");
  }
}

template <typename T>
void TensorOperationsModule<T>::add(const shared_ptr<const Tensor<T>> a,
                                    const shared_ptr<const Tensor<T>> b,
                                    shared_ptr<Tensor<T>> c) const {
  this.add_ptr(a, b, c);
}

template <typename T>
void TensorOperationsModule<T>::subtract(const shared_ptr<Tensor<T>> a,
                                         const shared_ptr<Tensor<T>> b,
                                         shared_ptr<Tensor<T>> c) const {
  this.subtract_ptr(a, b, c);
}

template <typename T>
void TensorOperationsModule<T>::multiply(const shared_ptr<Tensor<T>> a,
                                         const T b,
                                         shared_ptr<Tensor<T>> c) const {
  this.multiply_ptr(a, b, c);
}

template <typename T>
bool TensorOperationsModule<T>::equals(const shared_ptr<Tensor<T>> a,
                                       const shared_ptr<Tensor<T>> b) const {
  return this.equals_ptr(a, b);
}

template <typename T>
void TensorOperationsModule<T>::elementwise(
    const shared_ptr<const Tensor<T>> a, function<T(T)> f,
    const shared_ptr<Tensor<T>> c) const {
  this.elementwise_ptr(a, f, c);
}

template <typename T>
void TensorOperationsModule<T>::elementwise_in_place(
    const shared_ptr<Tensor<T>> a, function<T(T)> f) const {
  this.elementwise_in_place_ptr(a, f);
}

template <typename T>
void TensorOperationsModule<T>::gemm(int TA, int TB, int M, int N, int K,
                                     T ALPHA, shared_ptr<Tensor<T>> A, int lda,
                                     shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                     shared_ptr<Tensor<T>> C, int ldc) {
  this.gemm_ptr(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

template <typename T>
shared_ptr<Tensor<T>> TensorOperationsModule<T>::gemm_onnx(
    shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha, float beta,
    int transA, int transB, optional<shared_ptr<Tensor<T>>> C) {
  return this.gemm_onnx_ptr(A, B, alpha, beta, transA, transB, C);
}