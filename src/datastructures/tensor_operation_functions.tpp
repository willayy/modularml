#pragma once

#if defined(USE_AVX_GEMM) || defined(USE_AVX512_GEMM)
#include <immintrin.h>
#endif

#include "datastructures/mml_tensor.hpp"
#include "datastructures/tensor_operation_functions.hpp"

template <typename T>
static void mml_gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   std::shared_ptr<Tensor<T>> A, int lda,
                                   std::shared_ptr<Tensor<T>> B, int ldb,
                                   T BETA, std::shared_ptr<Tensor<T>> C,
                                   int ldc) {
  int k_col;
  int i_col_out;

  if (TA == 1) A->transpose();
  if (TB == 1) B->transpose();

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
static void mml_gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   std::shared_ptr<Tensor<T>> A, int lda,
                                   std::shared_ptr<Tensor<T>> B, int ldb,
                                   T BETA, std::shared_ptr<Tensor<T>> C,
                                   int ldc) {
  int i_col;
  int k_col;
  int i_col_out;

  if (TA == 1) A->transpose();
  if (TB == 1) B->transpose();

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
static void mml_gemm_row_wise_product(int TA, int TB, int M, int N, int K,
                                      T ALPHA, std::shared_ptr<Tensor<T>> A,
                                      int lda, std::shared_ptr<Tensor<T>> B,
                                      int ldb, T BETA,
                                      std::shared_ptr<Tensor<T>> C, int ldc) {
  int i_col;
  int k_col;
  int i_col_out;

  if (TA == 1) A->transpose();
  if (TB == 1) B->transpose();

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
static void mml_gemm_col_wise_product(int TA, int TB, int M, int N, int K,
                                      T ALPHA, std::shared_ptr<Tensor<T>> A,
                                      int lda, std::shared_ptr<Tensor<T>> B,
                                      int ldb, T BETA,
                                      std::shared_ptr<Tensor<T>> C, int ldc) {
  int i_col;
  int k_col;
  int i_col_out;

  if (TA == 1) A->transpose();
  if (TB == 1) B->transpose();

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

template <typename T>
static void mml_gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                             std::shared_ptr<Tensor<T>> A, int lda,
                             std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             std::shared_ptr<Tensor<T>> C, int ldc) {
  int block_size = 64;  // This depends on the CPU architecture - We can look
                        // into having the size of this be dynamically fetched
  if (!TA && !TB) {
    int i, j, jj, k, kk;
    int i_col, k_col, i_col_out;

    for (int jj = 0; jj < N; jj += block_size) {
      for (int kk = 0; kk < K; kk += block_size) {
        for (int i = 0; i < M; i++) {
          i_col = i * lda;
          i_col_out = i * ldc;
          for (int j = jj; j < std::min(jj + block_size, N); j++) {
            T acc = BETA * (*C)[i_col_out + j];
            for (int k = kk; k < std::min(kk + block_size, K); k++) {
              k_col = k * ldb;
              acc += ALPHA * (*A)[i_col + k] * (*B)[k_col + j];
            }
            (*C)[i_col_out + j] = acc;
          }
        }
      }
    }
  } else if (TA && !TB) {
    throw std::invalid_argument(
        "Transposition not yet supported in GEMM blocked.");
  } else if (!TA && TB) {
    throw std::invalid_argument(
        "Transposition not yet supported in GEMM blocked.");
  } else {
    throw std::invalid_argument(
        "Transposition not yet supported in GEMM blocked.");
  }
  return;
}

#ifdef USE_AVX_GEMM
template <typename T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         std::shared_ptr<Tensor<T>> A, int lda,
                         std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         std::shared_ptr<Tensor<T>> C, int ldc) {
  if (TA == 1) A->transpose();
  if (TB == 1) B->transpose();

  if constexpr (std::is_same<T, float>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 8) {
        __m256 c_val = _mm256_set1_ps((*C)[i * ldc + j]);
        __m256 sum = _mm256_setzero_ps();

        for (int k = 0; k < K; k++) {
          __m256 a_vals = _mm256_loadu_ps(&(*A)[i * lda + k]);

          __m256 b_vals = _mm256_loadu_ps(&(*B)[k * ldb + j]);

          sum = _mm256_fmadd_ps(a_vals, b_vals, sum);
        }

        sum = _mm256_fmadd_ps(sum, _mm256_set1_ps(ALPHA), c_val);
        sum = _mm256_add_ps(sum, _mm256_set1_ps(BETA));

        _mm256_storeu_ps(&(*C)[i * ldc + j], sum);
      }
    }
  } else if constexpr (std::is_same<T, double>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 4) {
        __m256d c_val = _mm256_set1_pd((*C)[i * ldc + j]);
        __m256d sum = _mm256_setzero_pd();

        for (int k = 0; k < K; k++) {
          __m256d a_vals = _mm256_loadu_pd(&(*A)[i * lda + k]);

          __m256d b_vals = _mm256_loadu_pd(&(*B)[k * ldb + j]);

          sum = _mm256_fmadd_pd(a_vals, b_vals, sum);
        }

        sum = _mm256_fmadd_pd(sum, _mm256_set1_pd(ALPHA), c_val);
        sum = _mm256_add_pd(sum, _mm256_set1_pd(BETA));

        _mm256_storeu_pd(&(*C)[i * ldc + j], sum);
      }
    }
  } else if constexpr (std::is_same<T, int>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 8) {
        __m256i sum = _mm256_setzero_si256();

        for (int k = 0; k < K; k++) {
          int a_scalar = (*A)[i * lda + k];
          __m256i a_broadcast = _mm256_set1_epi32(a_scalar);

          __m256i b_vals = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(&(*B)[k * ldb + j]));
          __m256i product = _mm256_mullo_epi32(a_broadcast, b_vals);

          sum = _mm256_add_epi32(sum, product);
        }

        sum = _mm256_mullo_epi32(sum, _mm256_set1_epi32(ALPHA));
        sum = _mm256_add_epi32(sum, _mm256_set1_epi32(BETA));

        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&(*C)[i * ldc + j]),
                            sum);
      }
    }
  } else {
    throw std::runtime_error("AVX2 only suppports double, float or int");
  }
  return;
}
#endif

#ifdef USE_AVX512_GEMM
template <typename T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            std::shared_ptr<Tensor<T>> A, int lda,
                            std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            std::shared_ptr<Tensor<T>> C, int ldc) {
  if (TA == 1)
    throw std::invalid_argument(
        "Transpose A not yet supported for AVX-512 GEMM.");
  if (TB == 1)
    throw std::invalid_argument(
        "Transpose B not yet supported for AVX-512 GEMM.");

  if constexpr (std::is_same<T, float>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 16) {
        __m512 c_val = _mm512_loadu_ps(&(*C)[i * ldc + j]);
        __m512 sum = _mm512_setzero_ps();

        for (int k = 0; k < K; k++) {
          __m512 a_vals = _mm512_set1_ps((*A)[i * lda + k]);

          __m512 b_vals = _mm512_loadu_ps(&(*B)[k * ldb + j]);

          sum = _mm512_fmadd_ps(a_vals, b_vals, sum);
        }

        sum = _mm512_fmadd_ps(_mm512_set1_ps(ALPHA), sum, c_val);
        sum = _mm512_add_ps(sum, _mm512_set1_ps(BETA));

        _mm512_storeu_ps(&(*C)[i * ldc + j], sum);
      }
    }
  } else if constexpr (std::is_same<T, double>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 8) {
        __m512d c_val = _mm512_loadu_pd(&(*C)[i * ldc + j]);
        __m512d sum = _mm512_setzero_pd();

        for (int k = 0; k < K; k++) {
          __m512d a_vals = _mm512_set1_pd((*A)[i * lda + k]);

          __m512d b_vals = _mm512_loadu_pd(&(*B)[k * ldb + j]);

          sum = _mm512_fmadd_pd(a_vals, b_vals, sum);
        }

        sum = _mm512_fmadd_pd(_mm512_set1_pd(ALPHA), sum, c_val);
        sum = _mm512_add_pd(sum, _mm512_set1_pd(BETA));

        _mm512_storeu_pd(&(*C)[i * ldc + j], sum);
      }
    }
  } else if constexpr (std::is_same<T, int>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 16) {
        __m512i sum = _mm512_setzero_si512();

        for (int k = 0; k < K; ++k) {
          __m512i a_vals =
              _mm512_set1_epi32((*A)[i * lda + k]);  // scalar broadcast

          __m512i b_vals = _mm512_loadu_si512(
              reinterpret_cast<const void *>(&(*B)[k * ldb + j]));

          __m512i product = _mm512_mullo_epi32(a_vals, b_vals);
          sum = _mm512_add_epi32(sum, product);
        }

        sum = _mm512_mullo_epi32(_mm512_set1_epi32(ALPHA), sum);
        sum = _mm512_add_epi32(sum, _mm512_set1_epi32(BETA));

        _mm512_storeu_si512(reinterpret_cast<void *>(&(*C)[i * ldc + j]), sum);
      }
    }
  } else {
    throw std::runtime_error("AVX-512 only suppports double, float or int");
  }
}
#endif

template <typename T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("Intel MKL GEMM not yet supported.");
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_inner_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_inner_product(transA, transB, M, N, K, static_cast<T>(alpha), A, lda,
                         B, ldb, static_cast<T>(beta), C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_outer_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_outer_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta,
                         C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_row_wise_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_row_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_col_wise_product(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_col_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_blocked(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_blocked(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                   ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_avx(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_avx(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_avx512(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_avx512(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                  ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>> mml_onnx_gemm_intel_MKL(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B, float alpha,
    float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  std::shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C
                    : TensorFactory::create_tensor<T>(
                          {static_cast<size_t>(M), static_cast<size_t>(N)});
  mml_gemm_intel(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                 ldc);
  return C_p;
}

template <typename T>
static void mml_add(const std::shared_ptr<const Tensor<T>> a,
                    const std::shared_ptr<const Tensor<T>> b,
                    std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
static void mml_subtract(const std::shared_ptr<Tensor<T>> a,
                         const std::shared_ptr<Tensor<T>> b,
                         std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
static void mml_multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                         std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
static bool mml_equals(const std::shared_ptr<Tensor<T>> a,
                       const std::shared_ptr<Tensor<T>> b) {
  if (a->get_size() != b->get_size() || a->get_shape() != b->get_shape()) {
    return false;
  } else {
    const auto size = a->get_size();
    for (size_t i = 0; i < size; i++) {
      if ((*a)[i] != (*b)[i]) {
        return false;
      }
    }
    return true;
  }
}

template <typename T>
static void mml_elementwise(const std::shared_ptr<const Tensor<T>> a,
                            const std::function<T(T)> &f,
                            const std::shared_ptr<Tensor<T>> c) {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<size_t> indices(num_dimensions);
  for (size_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = a->get_size();

  for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply std::function `f` from `a` to `c`
    (*c)[indices] = f((*a)[indices]);

    // Increment indices
    size_t d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break;  // No carry needed, continue iteration
      }
      indices[d] = 0;  // Carry over to the next dimension
    } while (d-- > 0);
  }
}

template <typename T>
static void mml_elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                     const std::function<T(T)> &f) {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<size_t> indices(num_dimensions);
  for (size_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }

  const auto total_elements = a->get_size();

  for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply the std::function `f` to the current element
    (*a)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    size_t d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break;  // No carry needed, continue iteration
      }
      indices[d] = 0;  // Carry over to the next dimension
    } while (d-- > 0);
  }
}

template <typename T>
static int mml_arg_max(const std::shared_ptr<const Tensor<T>> a) {
  const auto size = a->get_size();
  if (size == 0) {
    throw std::runtime_error("arg_max called on an empty tensor.");
  }

  T max_value = (*a)[0];
  int max_index = 0;

  for (int i = 1; i < static_cast<int>(size); ++i) {
    if ((*a)[i] > max_value) {
      max_value = (*a)[i];
      max_index = i;
    }
  }

  return max_index;
}

template <typename T>
static void mml_sliding_window(
    const array_mml<size_t> &in_shape, const array_mml<size_t> &out_shape,
    const std::vector<int> &kernel_shape, const std::vector<int> &strides,
    const std::vector<int> &dilations,
    const std::vector<std::pair<int, int>> &pads,
    const std::function<void(const std::vector<std::vector<size_t>> &,
                             const std::vector<size_t> &)> &window_f) {
  size_t total_rank = in_shape.size();
  size_t spatial_rank = kernel_shape.size();

  std::vector<size_t> out_idx(total_rank, 0);

  std::function<void(size_t)> recurse = [&](size_t dim) {
    if (dim == total_rank) {  // Depth reached

      std::vector<std::vector<size_t>> window_in_idx;
      std::vector<int> kernel_pos(spatial_rank, 0);

      std::function<void(size_t)> kernel_recurse = [&](size_t kdim) {
        if (kdim == spatial_rank) {  // Depth reached
          bool valid = true;
          std::vector<size_t> in_idx(total_rank, 0);
          in_idx[0] = out_idx[0];  // Batch
          in_idx[1] = out_idx[1];  // Channel

          for (size_t i = 0; i < spatial_rank; ++i) {
            int out_coord = static_cast<int>(out_idx[i + 2]);
            int start = out_coord * strides[i] - pads[i].first;
            int offset = kernel_pos[i] * dilations[i];
            int pos = start + offset;

            if (pos < 0 || pos >= static_cast<int>(in_shape[i + 2])) {
              valid = false;
              break;
            }
            in_idx[i + 2] = static_cast<size_t>(pos);
          }

          if (valid) {
            window_in_idx.push_back(in_idx);
          }
          return;
        }

        for (int k = 0; k < kernel_shape[kdim]; ++k) {
          kernel_pos[kdim] = k;
          kernel_recurse(kdim + 1);
        }
      };
      kernel_recurse(0);

      window_f(window_in_idx, out_idx);
      return;
    }

    for (size_t i = 0; i < out_shape[dim]; ++i) {
      out_idx[dim] = i;
      recurse(dim + 1);
    }
  };

  recurse(0);
}
