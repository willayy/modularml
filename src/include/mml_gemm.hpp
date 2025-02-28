#pragma once

#include "a_gemm.hpp"
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
                          shared_ptr<Tensor<T>> C, int ldc) override {
    if (!TA && !TB) {
      int i, j, k;
      int i_col, k_col, i_col_out;
      T acc;

      for (i = 0; i < M; i++) {
        i_col = i * lda;
        i_col_out = i * ldc;
        for (j = 0; j < N; j++) {
          acc = ((T)BETA) * (*C)[i_col_out + j];
          for (k = 0; k < K; k++) {
            k_col = k * ldb;
            acc += ((T)ALPHA) * (*A)[i_col + k] * (*B)[k_col + j];
          }
          (*C)[i_col_out + j] = acc;
        }
      }
    } else {
      logic_error("Transposition not yet supported in GEMM inner product.");
    }
    return;
  }

  void gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                          shared_ptr<Tensor<T>> A, int lda,
                          shared_ptr<Tensor<T>> B, int ldb,
                          T BETA,
                          shared_ptr<Tensor<T>> C, int ldc) override {
    if (!TA && !TB) {
      int i, j, k;
      int i_col, k_col, i_col_out;

      for (i = 0; i < M; i++) {
        i_col_out = i * ldc;
        for (j = 0; j < N; j++) {
          (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
        }
      }

      for (k = 0; k < K; k++) {
        k_col = k * ldb;
        for (i = 0; i < M; i++) {
          i_col = i * lda;
          i_col_out = i * ldc;
          for (j = 0; j < N; j++) {
            (*C)[i_col_out + j] += ((T)ALPHA) * (*A)[i_col + k] * (*B)[k_col + j];
          }
        }
      }
    } else {
      logic_error("Transposition not yet supported in GEMM outer product.");
    }
    return;
  }

  void gemm_row_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             shared_ptr<Tensor<T>> A, int lda,
                             shared_ptr<Tensor<T>> B, int ldb,
                             T BETA,
                             shared_ptr<Tensor<T>> C, int ldc) override {
    if (!TA && !TB) {
      int i, j, k;
      int i_col, k_col, i_col_out;

      for (i = 0; i < M; i++) {
        i_col = i * lda;
        i_col_out = i * ldc;
        for (j = 0; j < N; j++) {
          (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
        }
        for (k = 0; k < K; k++) {
          k_col = k * ldb;
          for (j = 0; j < N; j++) {
            (*C)[i_col_out + j] += ((T)ALPHA) * (*A)[i_col + k] * (*B)[k_col + j];
          }
        }
      }
    } else {
      logic_error("Transposition not yet supported in GEMM row-wise product.");
    }
    return;
  }

  void gemm_col_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             shared_ptr<Tensor<T>> A, int lda,
                             shared_ptr<Tensor<T>> B, int ldb,
                             T BETA,
                             shared_ptr<Tensor<T>> C, int ldc) override {
    if (!TA && !TB) {
      int i, j, k;
      int i_col, k_col, i_col_out;

      for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
          i_col_out = i * ldc;
          (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
        }
        for (k = 0; k < K; k++) {
          k_col = k * ldb;
          for (i = 0; i < M; i++) {
            i_col = i * lda;
            i_col_out = i * ldc;
            (*C)[i_col_out + j] += ((T)ALPHA) * (*A)[i_col + k] * (*B)[k_col + j];
          }
        }
      }
    } else {
      logic_error("Transposition not yet supported in GEMM col-wise product.");
    }
    return;
  }

  shared_ptr<GemmModule<T>> clone() const override {
    return make_shared<Gemm_mml>(*this);
  }

#pragma GCC diagnostic ignored "-Wunused-parameter"
  void gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                    shared_ptr<Tensor<T>> A, int lda,
                    shared_ptr<Tensor<T>> B, int ldb,
                    T BETA,
                    shared_ptr<Tensor<T>> C, int ldc) override {
    logic_error("Blocked GEMM not yet supported.");
  }

  void gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                shared_ptr<Tensor<T>> A, int lda,
                shared_ptr<Tensor<T>> B, int ldb,
                T BETA,
                shared_ptr<Tensor<T>> C, int ldc) override {
    logic_error("AVX GEMM not yet supported.");
  }

  void gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                   shared_ptr<Tensor<T>> A, int lda,
                   shared_ptr<Tensor<T>> B, int ldb,
                   T BETA,
                   shared_ptr<Tensor<T>> C, int ldc) override {
    logic_error("AVX-512 GEMM not yet supported.");
  }

  void gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                      shared_ptr<Tensor<T>> A, int lda,
                      shared_ptr<Tensor<T>> B, int ldb,
                      T BETA,
                      shared_ptr<Tensor<T>> C, int ldc) override {
    logic_error("Intel MKL GEMM not yet supported.");
  }
};