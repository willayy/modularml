#pragma once
#include "a_tensor.hpp"
#include "tensor_factory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

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
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_outer_product(
    std::shared_ptr<Tensor<T>> A = nullptr,
    std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0, float beta = 1.0,
    int transA = 0, int transB = 0,
    std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_row_wise_product(
    std::shared_ptr<Tensor<T>> A = nullptr,
    std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0, float beta = 1.0,
    int transA = 0, int transB = 0,
    std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_col_wise_product(
    std::shared_ptr<Tensor<T>> A = nullptr,
    std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0, float beta = 1.0,
    int transA = 0, int transB = 0,
    std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_blocked(
    std::shared_ptr<Tensor<T>> A = nullptr,
    std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0, float beta = 1.0,
    int transA = 0, int transB = 0,
    std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_avx(std::shared_ptr<Tensor<T>> A = nullptr,
                  std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
                  float beta = 1.0, int transA = 0, int transB = 0,
                  std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_avx512(
    std::shared_ptr<Tensor<T>> A = nullptr,
    std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0, float beta = 1.0,
    int transA = 0, int transB = 0,
    std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_intel_MKL(
    std::shared_ptr<Tensor<T>> A = nullptr,
    std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0, float beta = 1.0,
    int transA = 0, int transB = 0,
    std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

template <typename T>
static void mml_gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   std::shared_ptr<Tensor<T>> A, int lda,
                                   std::shared_ptr<Tensor<T>> B, int ldb,
                                   T BETA, std::shared_ptr<Tensor<T>> C,
                                   int ldc);

template <typename T>
static void mml_gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   std::shared_ptr<Tensor<T>> A, int lda,
                                   std::shared_ptr<Tensor<T>> B, int ldb,
                                   T BETA, std::shared_ptr<Tensor<T>> C,
                                   int ldc);

template <typename T>
static void mml_gemm_row_wise_product(int TA, int TB, int M, int N, int K,
                                      T ALPHA, std::shared_ptr<Tensor<T>> A,
                                      int lda, std::shared_ptr<Tensor<T>> B,
                                      int ldb, T BETA,
                                      std::shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_col_wise_product(int TA, int TB, int M, int N, int K,
                                      T ALPHA, std::shared_ptr<Tensor<T>> A,
                                      int lda, std::shared_ptr<Tensor<T>> B,
                                      int ldb, T BETA,
                                      std::shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                             std::shared_ptr<Tensor<T>> A, int lda,
                             std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             std::shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         std::shared_ptr<Tensor<T>> A, int lda,
                         std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         std::shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            std::shared_ptr<Tensor<T>> A, int lda,
                            std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            std::shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               std::shared_ptr<Tensor<T>> C, int ldc);

template <typename T>
static void mml_add(const std::shared_ptr<const Tensor<T>> a,
                    const std::shared_ptr<const Tensor<T>> b,
                    std::shared_ptr<Tensor<T>> c);

template <typename T>
static void mml_subtract(const std::shared_ptr<Tensor<T>> a,
                         const std::shared_ptr<Tensor<T>> b,
                         std::shared_ptr<Tensor<T>> c);

template <typename T>
static void mml_multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                         std::shared_ptr<Tensor<T>> c);

template <typename T>
static bool mml_equals(const std::shared_ptr<Tensor<T>> a,
                       const std::shared_ptr<Tensor<T>> b);

template <typename T>
static void mml_elementwise(const std::shared_ptr<const Tensor<T>> a,
                            const std::function<T(T)> &f,
                            const std::shared_ptr<Tensor<T>> c);

template <typename T>
static void mml_elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                     const std::function<T(T)> &f);

template <typename T>
static int mml_arg_max(const std::shared_ptr<const Tensor<T>> a);

template <typename T>
static void mml_sliding_window(
  const std::shared_ptr<const Tensor<T>>& input,
  std::shared_ptr<Tensor<T>>& output,
  const std::optional<std::shared_ptr<Tensor<int64_t>>>& indices_out,
  const std::vector<int>& kernel_shape,
  const std::vector<int>& strides,
  const std::vector<int>& dilations,
  const std::vector<std::pair<int, int>>& pads,
  const int storage_order,
  const function<T(const std::vector<T>&, const std::vector<int64_t>&, int64_t&)> &window_f
);

#include "../datastructures/tensor_operation_functions.tpp"