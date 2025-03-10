#pragma once

#include "include/mml_gemm.hpp"


template <typename T>
void check_inputs(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, shared_ptr<Tensor<T>> C, int TA, int TB, int M, int N, int K, int lda, int ldb, int ldc) {
  if (TA) {
    if (lda < K) {
      logic_error("lda must be >= K when TA is true.");
    }
  } else {
    if (lda < M) {
      logic_error("lda must be >= M when TA is false.");
    }
  }
  if (TB) {
    if (ldb < N) {
      logic_error("ldb must be >= N when TB is true.");
    }
  } else {
    if (ldb < K) {
      logic_error("ldb must be >= K when TB is false.");
    }
  }
  if (ldc < N) {
    logic_error("ldc must be >= N.");
  }
  if (M < 0 || N < 0 || K < 0) {
    logic_error("M, N, and K must be >= 0.");
  }
  if (TA && TB) {
    if (M != N) {
      logic_error("M and N must be equal when TA and TB are both true.");
    }
  }
  if ((*A).get_size() != M * K) {
    logic_error("Size of A must be M * K.");
  }
  if ((*B).get_size() != K * N) {
    logic_error("Size of B must be K * N.");
  }
  if ((*C).get_size() != M * N) {
    logic_error("Size of C must be M * N.");
  }
  if ((*A).is_matrix() == false) {
    logic_error("A must be a matrix.");
  }
  if ((*B).is_matrix() == false) {
    logic_error("B must be a matrix.");
  }
  if ((*C).is_matrix() == false) {
    logic_error("C must be a matrix.");
  }
  if ((*A).matrix_match((*B)) == false) {
    logic_error("A and B must have matching matrix dimensions.");
  }
  if ((*C).get_shape()[0] != M || (*C).get_shape()[1] != N) {
    logic_error("C must have dimensions M x N.");
  }
}

template <typename T>
void Gemm_mml<T>::gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                     shared_ptr<Tensor<T>> A, int lda,
                                     shared_ptr<Tensor<T>> B, int ldb,
                                     T BETA,
                                     shared_ptr<Tensor<T>> C, int ldc) {
  check_inputs(A, B, C, TA, TB, M, N, K, lda, ldb, ldc);
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
