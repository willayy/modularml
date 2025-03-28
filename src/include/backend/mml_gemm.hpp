#pragma once

#include "backend/a_gemm.hpp"
#include "globals.hpp"

template <typename T>
class Gemm_mml : public GemmModule<T> {
 public:
  /// @brief Default constructor for GEMM_mml class.
  Gemm_mml() = default;

  /// @brief Copy constructor for GEMM_mml class.
  Gemm_mml(const Gemm_mml& other) = default;

  /// @brief Move constructor for GEMM_mml class.
  Gemm_mml(Gemm_mml&& other) noexcept = default;

  /// @brief Destructor for GEMM_mml class.
  ~Gemm_mml() override = default;

  void gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                          shared_ptr<Tensor<T>> A, int lda,
                          shared_ptr<Tensor<T>> B, int ldb,
                          T BETA,
                          shared_ptr<Tensor<T>> C, int ldc) override;

  void gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                          shared_ptr<Tensor<T>> A, int lda,
                          shared_ptr<Tensor<T>> B, int ldb,
                          T BETA,
                          shared_ptr<Tensor<T>> C, int ldc) override;

  void gemm_row_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             shared_ptr<Tensor<T>> A, int lda,
                             shared_ptr<Tensor<T>> B, int ldb,
                             T BETA,
                             shared_ptr<Tensor<T>> C, int ldc) override;

  void gemm_col_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             shared_ptr<Tensor<T>> A, int lda,
                             shared_ptr<Tensor<T>> B, int ldb,
                             T BETA,
                             shared_ptr<Tensor<T>> C, int ldc) override;

  void gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                    shared_ptr<Tensor<T>> A, int lda,
                    shared_ptr<Tensor<T>> B, int ldb,
                    T BETA,
                    shared_ptr<Tensor<T>> C, int ldc) override;

  void gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                shared_ptr<Tensor<T>> A, int lda,
                shared_ptr<Tensor<T>> B, int ldb,
                T BETA,
                shared_ptr<Tensor<T>> C, int ldc) override;

  void gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                   shared_ptr<Tensor<T>> A, int lda,
                   shared_ptr<Tensor<T>> B, int ldb,
                   T BETA,
                   shared_ptr<Tensor<T>> C, int ldc) override;

  void gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                      shared_ptr<Tensor<T>> A, int lda,
                      shared_ptr<Tensor<T>> B, int ldb,
                      T BETA,
                      shared_ptr<Tensor<T>> C, int ldc) override;
};

#include "../backend/mml_gemm.tpp"