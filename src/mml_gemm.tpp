#pragma once

#include "include/mml_gemm.hpp"

template <typename T>
void Gemm_mml<T>::gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                     shared_ptr<Tensor<T>> A, int lda,
                                     shared_ptr<Tensor<T>> B, int ldb,
                                     T BETA,
                                     shared_ptr<Tensor<T>> C, int ldc) {
  if (!TA && !TB) {
    int i, j, k;
    int i_col, k_col, i_col_out;

    for (i = 0; i < M; i++) {
      i_col_out = i * ldc;
      for (j = 0; j < N; j++) {
        (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
      }
      for (k = 0; k < K; k++) {
        k_col = k * ldb;
        for (j = 0; j < N; j++) {
          (*C)[i_col_out + j] += ((T)ALPHA) * (*A)[i * lda + k] * (*B)[k_col + j];
        }
      }
    }
  } else {
    logic_error("Transposition not yet supported in GEMM inner product.");
  }
  return;
}

template <typename T>
void Gemm_mml<T>::gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                     shared_ptr<Tensor<T>> A, int lda,
                                     shared_ptr<Tensor<T>> B, int ldb,
                                     T BETA,
                                     shared_ptr<Tensor<T>> C, int ldc) {
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

template <typename T>
void Gemm_mml<T>::gemm_row_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                        shared_ptr<Tensor<T>> A, int lda,
                                        shared_ptr<Tensor<T>> B, int ldb,
                                        T BETA,
                                        shared_ptr<Tensor<T>> C, int ldc) {
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

template <typename T>
void Gemm_mml<T>::gemm_col_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                        shared_ptr<Tensor<T>> A, int lda,
                                        shared_ptr<Tensor<T>> B, int ldb,
                                        T BETA,
                                        shared_ptr<Tensor<T>> C, int ldc) {
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

#pragma GCC diagnostic ignored "-Wunused-parameter"
template <typename T>
void Gemm_mml<T>::gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                               shared_ptr<Tensor<T>> A, int lda,
                               shared_ptr<Tensor<T>> B, int ldb,
                               T BETA,
                               shared_ptr<Tensor<T>> C, int ldc) {
  logic_error("Blocked GEMM not yet supported.");
}
template <typename T>
void Gemm_mml<T>::gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                           shared_ptr<Tensor<T>> A, int lda,
                           shared_ptr<Tensor<T>> B, int ldb,
                           T BETA,
                           shared_ptr<Tensor<T>> C, int ldc) {
  logic_error("AVX GEMM not yet supported.");
}
template <typename T>
void Gemm_mml<T>::gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                              shared_ptr<Tensor<T>> A, int lda,
                              shared_ptr<Tensor<T>> B, int ldb,
                              T BETA,
                              shared_ptr<Tensor<T>> C, int ldc) {
  logic_error("AVX-512 GEMM not yet supported.");
}
template <typename T>
void Gemm_mml<T>::gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                                 shared_ptr<Tensor<T>> A, int lda,
                                 shared_ptr<Tensor<T>> B, int ldb,
                                 T BETA,
                                 shared_ptr<Tensor<T>> C, int ldc) {
  logic_error("Intel MKL GEMM not yet supported.");
}
