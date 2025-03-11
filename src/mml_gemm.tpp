#pragma once

#include "include/mml_gemm.hpp"

void static check_lda(int TA, int lda, int M, int K) {
  if (TA) {
    if (lda < K) {
      logic_error("lda must be >= K when TA is true.");
    }
  } else {
    if (lda < M) {
      logic_error("lda must be >= M when TA is false.");
    }
  }
}

void static check_ldb(int TB, int ldb, int K, int N) {
  if (TB) {
    if (ldb < N) {
      logic_error("ldb must be >= N when TB is true.");
    }
  } else {
    if (ldb < K) {
      logic_error("ldb must be >= K when TB is false.");
    }
  }
}

void static check_ldc(int ldc, int N) {
  if (ldc < N) {
    logic_error("ldc must be >= N.");
  }
}

void static check_dimensions(int M, int N, int K) {
  if (M < 0 || N < 0 || K < 0) {
    logic_error("M, N, and K must be >= 0.");
  }
}

template <typename T>
void static check_tensor_sizes(shared_ptr<Tensor<T>> A,
                               shared_ptr<Tensor<T>> B,
                               shared_ptr<Tensor<T>> C,
                               int M, int N, int K) {
  if ((*A).get_size() != M * K) {
    logic_error("Size of A must be M * K.");
  }
  if ((*B).get_size() != K * N) {
    logic_error("Size of B must be K * N.");
  }
  if ((*C).get_size() != M * N) {
    logic_error("Size of C must be M * N.");
  }
}

template <typename T>
void static check_tensor_properties(shared_ptr<Tensor<T>> A,
                                    shared_ptr<Tensor<T>> B,
                                    shared_ptr<Tensor<T>> C) {
  if (!(*A).is_matrix()) {
    logic_error("A must be a matrix.");
  }
  if (!(*B).is_matrix()) {
    logic_error("B must be a matrix.");
  }
  if (!(*C).is_matrix()) {
    logic_error("C must be a matrix.");
  }
}

template <typename T>
void static check_matrix_match(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B) {
  if (!(*A).matrix_match((*B))) {
    logic_error("A and B must have matching matrix dimensions.");
  }
}

template <typename T>
void static check_C_dimensions(shared_ptr<Tensor<T>> C, int M, int N) {
  if ((*C).get_shape()[0] != M || (*C).get_shape()[1] != N) {
    logic_error("C must have dimensions M x N.");
  }
}

template <typename T>
void static check_inputs(shared_ptr<Tensor<T>> A,
                         shared_ptr<Tensor<T>> B,
                         shared_ptr<Tensor<T>> C,
                         int TA, int TB, int M,
                         int N, int K, int lda,
                         int ldb, int ldc) {
  check_lda(TA, lda, M, K);
  check_ldb(TB, ldb, K, N);
  check_ldc(ldc, N);
  check_dimensions(M, N, K);
  check_tensor_sizes(A, B, C, M, N, K);
  check_tensor_properties(A, B, C);
  check_matrix_match(A, B);
  check_C_dimensions(C, M, N);
}

template <typename T>
void Gemm_mml<T>::gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                     shared_ptr<Tensor<T>> A, int lda,
                                     shared_ptr<Tensor<T>> B, int ldb,
                                     T BETA,
                                     shared_ptr<Tensor<T>> C, int ldc) {
  check_inputs(A, B, C, TA, TB, M, N, K, lda, ldb, ldc);
  if (!TA && !TB) {
    int k_col;
    int i_col_out;

    for (int i = 0; i < M; i++) {
      i_col_out = i * ldc;

      for (int j = 0; j < N; j++) {
        (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
      }
      for (int k = 0; k < K; k++) {
        k_col = k * ldb;

        for (int j = 0; j < N; j++) {
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
  check_inputs(A, B, C, TA, TB, M, N, K, lda, ldb, ldc);
  if (!TA && !TB) {
    int i_col;
    int k_col;
    int i_col_out;

    for (int i = 0; i < M; i++) {
      i_col_out = i * ldc;

      for (int j = 0; j < N; j++) {
        (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
      }
    }

    for (int k = 0; k < K; k++) {
      k_col = k * ldb;

      for (int i = 0; i < M; i++) {
        i_col = i * lda;
        i_col_out = i * ldc;

        for (int j = 0; j < N; j++) {
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
  check_inputs(A, B, C, TA, TB, M, N, K, lda, ldb, ldc);
  if (!TA && !TB) {
    int i_col;
    int k_col;
    int i_col_out;

    for (int i = 0; i < M; i++) {
      i_col = i * lda;
      i_col_out = i * ldc;

      for (int j = 0; j < N; j++) {
        (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
      }

      for (int k = 0; k < K; k++) {
        k_col = k * ldb;

        for (int j = 0; j < N; j++) {
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
  check_inputs(A, B, C, TA, TB, M, N, K, lda, ldb, ldc);
  if (!TA && !TB) {
    int i_col;
    int k_col;
    int i_col_out;

    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        i_col_out = i * ldc;
        (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
      }

      for (int k = 0; k < K; k++) {
        k_col = k * ldb;

        for (int i = 0; i < M; i++) {
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
