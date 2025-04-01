#pragma once
#include "a_tensor.hpp"
#include "globals.hpp"
#include "tensor_factory.hpp"

/**
 * Standard Tensor operation functions that gets shipped with ModularML as
 * defaults.
 */

template <typename T>
static shared_ptr<Tensor<T>> mml_onnx_gemm_inner_product(
    shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
    float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
    optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static shared_ptr<Tensor<T>> mml_onnx_gemm_outer_product(
    shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
    float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
    optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static shared_ptr<Tensor<T>> mml_onnx_gemm_row_wise_product(
    shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
    float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
    optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static shared_ptr<Tensor<T>> mml_onnx_gemm_col_wise_product(
    shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
    float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
    optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_blocked(shared_ptr<Tensor<T>> A = nullptr,
                      shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
                      float beta = 1.0, int transA = 0, int transB = 0,
                      optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_avx(shared_ptr<Tensor<T>> A = nullptr,
                  shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
                  float beta = 1.0, int transA = 0, int transB = 0,
                  optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_avx512(shared_ptr<Tensor<T>> A = nullptr,
                     shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
                     float beta = 1.0, int transA = 0, int transB = 0,
                     optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_intel_MKL(shared_ptr<Tensor<T>> A = nullptr,
                        shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
                        float beta = 1.0, int transA = 0, int transB = 0,
                        optional<shared_ptr<Tensor<T>>> C = nullopt);

template <typename T>
static void mml_gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   shared_ptr<Tensor<T>> A, int lda,
                                   shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                   shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   shared_ptr<Tensor<T>> A, int lda,
                                   shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                   shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_row_wise_product(int TA, int TB, int M, int N, int K,
                                      T ALPHA, shared_ptr<Tensor<T>> A, int lda,
                                      shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                      shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_col_wise_product(int TA, int TB, int M, int N, int K,
                                      T ALPHA, shared_ptr<Tensor<T>> A, int lda,
                                      shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                      shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                             shared_ptr<Tensor<T>> A, int lda,
                             shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         shared_ptr<Tensor<T>> A, int lda,
                         shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            shared_ptr<Tensor<T>> A, int lda,
                            shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               shared_ptr<Tensor<T>> A, int lda,
                               shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_add(const shared_ptr<const Tensor<T>> a,
                    const shared_ptr<const Tensor<T>> b,
                    shared_ptr<Tensor<T>> c);

template <typename T>
static void mml_subtract(const shared_ptr<Tensor<T>> a,
                         const shared_ptr<Tensor<T>> b,
                         shared_ptr<Tensor<T>> c);

template <typename T>
static void mml_multiply(const shared_ptr<Tensor<T>> a, const T b,
                         shared_ptr<Tensor<T>> c);

template <typename T>
static bool mml_equals(const shared_ptr<Tensor<T>> a,
                       const shared_ptr<Tensor<T>> b);

template <typename T>
static void mml_elementwise(const shared_ptr<const Tensor<T>> a, 
                            const function<T(T)> &f,
                            const shared_ptr<Tensor<T>> c);

template <typename T>
static void mml_elementwise_in_place(const shared_ptr<Tensor<T>> a, 
                                     const function<T(T)> &f);

template <typename T>
static int mml_arg_max(const shared_ptr<const Tensor<T>> a);

#include "../datastructures/tensor_operation_functions.tpp"