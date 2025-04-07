#pragma once

#include "backend/a_gemm.hpp"
#include "../utility/uli.hpp"
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

template <typename T> class Gemm_mml : public GemmModule<T> {
public:
  /// @brief Default constructor for GEMM_mml class.
  [[deprecated("Use TensorOperationsModule instead")]]
  Gemm_mml() = default;

  /// @brief Copy constructor for GEMM_mml class.
  [[deprecated("Use TensorOperationsModule instead")]]
  Gemm_mml(const Gemm_mml &other) = default;

  /// @brief Move constructor for GEMM_mml class.
  [[deprecated("Use TensorOperationsModule instead")]]
  Gemm_mml(Gemm_mml &&other) noexcept = default;

  /// @brief Destructor for GEMM_mml class.
  [[deprecated("Use TensorOperationsModule instead")]]
  ~Gemm_mml() override = default;

  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                          std::shared_ptr<Tensor<T>> A, int lda,
                          std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                          std::shared_ptr<Tensor<T>> C, int ldc) override;

  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                          std::shared_ptr<Tensor<T>> A, int lda,
                          std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                          std::shared_ptr<Tensor<T>> C, int ldc) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_row_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             std::shared_ptr<Tensor<T>> A, int lda,
                             std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             std::shared_ptr<Tensor<T>> C, int ldc) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_col_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             std::shared_ptr<Tensor<T>> A, int lda,
                             std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             std::shared_ptr<Tensor<T>> C, int ldc) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                    std::shared_ptr<Tensor<T>> A, int lda,
                    std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                    std::shared_ptr<Tensor<T>> C, int ldc) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                std::shared_ptr<Tensor<T>> A, int lda,
                std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                std::shared_ptr<Tensor<T>> C, int ldc) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                   std::shared_ptr<Tensor<T>> A, int lda,
                   std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                   std::shared_ptr<Tensor<T>> C, int ldc) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                      std::shared_ptr<Tensor<T>> A, int lda,
                      std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                      std::shared_ptr<Tensor<T>> C, int ldc) override;
};

#include "../backend/mml_gemm.tpp"