#pragma once

#include "backend/mml_gemm.hpp"

void static check_lda(int lda, int K) {
  if (lda < K) {
    throw std::invalid_argument("lda must be >= K.");
  }
}

void static check_ldb(int ldb, int N) {
  if (ldb < N) {
    throw std::invalid_argument("ldb must be >= N.");
  }
}

void static check_ldc(int ldc, int N) {
  if (ldc < N) {
    throw std::invalid_argument("ldc must be >= N.");
  }
}

void static check_dimensions(int M, int N, int K) {
  if (M < 0 || N < 0 || K < 0) {
    throw std::invalid_argument("M, N, and K must be >= 0.");
  }
}

template <typename T>
void static check_tensor_sizes(std::shared_ptr<Tensor<T>> A,
                               std::shared_ptr<Tensor<T>> B,
                               std::shared_ptr<Tensor<T>> C, int M, int N,
                               int K) {
  if ((*A).get_size() != M * K) {
    throw std::invalid_argument("Size of A must be M * K.");
  }
  if ((*B).get_size() != K * N) {
    throw std::invalid_argument("Size of B must be K * N.");
  }
  if ((*C).get_size() != M * N) {
    throw std::invalid_argument("Size of C must be M * N.");
  }
}

template <typename T>
void static check_tensor_properties(std::shared_ptr<Tensor<T>> A,
                                    std::shared_ptr<Tensor<T>> B,
                                    std::shared_ptr<Tensor<T>> C) {

  if (!(*A).is_matrix()) {
    throw std::invalid_argument("A must be a matrix.");
  }
  if (!(*B).is_matrix()) {
    throw std::invalid_argument("B must be a matrix.");
  }
  if (!(*C).is_matrix()) {
    throw std::invalid_argument("C must be a matrix.");
  }
}

template <typename T>
void static check_matrix_match(std::shared_ptr<Tensor<T>> A,
                               std::shared_ptr<Tensor<T>> B, int TA, int TB) {
  if (!(*A).matrix_match((*B)) && !TA && !TB) {
    throw std::invalid_argument(
        "A and B must have matching matrix dimensions.");
  }
  if (TA && (*A).get_shape()[0] != (*B).get_shape()[0]) {
    throw std::invalid_argument(
        "A and B must have matching matrix dimensions.");
  }
  if (TB && (*A).get_shape()[1] != (*B).get_shape()[1]) {
    throw std::invalid_argument(
        "A and B must have matching matrix dimensions.");
  }
  if (TA && TB && (*A).get_shape()[0] != (*B).get_shape()[1]) {
    throw std::invalid_argument(
        "A and B must have matching matrix dimensions.");
  }
}

template <typename T>
void static check_C_dimensions(std::shared_ptr<Tensor<T>> C, int M, int N) {
  if ((*C).get_shape()[0] != M || (*C).get_shape()[1] != N) {
    throw std::invalid_argument("C must have dimensions M x N.");
  }
}

template <typename T>
void static check_inputs(int TA, std::shared_ptr<Tensor<T>> A, int TB,
                         std::shared_ptr<Tensor<T>> B,
                         std::shared_ptr<Tensor<T>> C, int M, int N, int K,
                         int lda, int ldb, int ldc) {
  check_lda(lda, K);
  check_ldb(ldb, N);
  check_ldc(ldc, N);
  check_dimensions(M, N, K);
  check_tensor_sizes(A, B, C, M, N, K);
  check_tensor_properties(A, B, C);
  check_matrix_match(A, B, TA, TB);
  check_C_dimensions(C, M, N);
}

template <typename T>
void Gemm_mml<T>::gemm_inner_product(int TA, int TB, int M, int N, int K,
                                     T ALPHA, std::shared_ptr<Tensor<T>> A,
                                     int lda, std::shared_ptr<Tensor<T>> B,
                                     int ldb, T BETA,
                                     std::shared_ptr<Tensor<T>> C, int ldc) {

  check_inputs(TA, A, TB, B, C, M, N, K, lda, ldb, ldc);

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

  return;
}

template <typename T>
void Gemm_mml<T>::gemm_outer_product(int TA, int TB, int M, int N, int K,
                                     T ALPHA, std::shared_ptr<Tensor<T>> A,
                                     int lda, std::shared_ptr<Tensor<T>> B,
                                     int ldb, T BETA,
                                     std::shared_ptr<Tensor<T>> C, int ldc) {

  check_inputs(TA, A, TB, B, C, M, N, K, lda, ldb, ldc);

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

  return;
}

template <typename T>
void Gemm_mml<T>::gemm_row_wise_product(int TA, int TB, int M, int N, int K,
                                        T ALPHA, std::shared_ptr<Tensor<T>> A,
                                        int lda, std::shared_ptr<Tensor<T>> B,
                                        int ldb, T BETA,
                                        std::shared_ptr<Tensor<T>> C, int ldc) {

  check_inputs(TA, A, TB, B, C, M, N, K, lda, ldb, ldc);

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

  return;
}

template <typename T>
void Gemm_mml<T>::gemm_col_wise_product(int TA, int TB, int M, int N, int K,
                                        T ALPHA, std::shared_ptr<Tensor<T>> A,
                                        int lda, std::shared_ptr<Tensor<T>> B,
                                        int ldb, T BETA,
                                        std::shared_ptr<Tensor<T>> C, int ldc) {

  check_inputs(TA, A, TB, B, C, M, N, K, lda, ldb, ldc);

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

  return;
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
template <typename T>
void Gemm_mml<T>::gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                               std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("Blocked GEMM not yet supported.");
}
template <typename T>
void Gemm_mml<T>::gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                           std::shared_ptr<Tensor<T>> A, int lda,
                           std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                           std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("AVX GEMM not yet supported.");
}
template <typename T>
void Gemm_mml<T>::gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                              std::shared_ptr<Tensor<T>> A, int lda,
                              std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                              std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("AVX-512 GEMM not yet supported.");
}
template <typename T>
void Gemm_mml<T>::gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                                 std::shared_ptr<Tensor<T>> A, int lda,
                                 std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                 std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("Intel MKL GEMM not yet supported.");
}
