#pragma once

#if defined(USE_AVX_GEMM) || defined(USE_AVX512_GEMM)
#include <immintrin.h>
#include "utility/avx_mask_helper.hpp"
#endif

#if defined(USE_OPENBLAS_GEMM)
#include <cblas.h>
#include <openblas_config.h>

#include <thread>
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
  int block_size = 32;  // Can be tuned or made adaptive later

  if (!TA && !TB) {
    for (int ii = 0; ii < M; ii += block_size) {
      for (int kk = 0; kk < K; kk += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
          for (int i = ii; i < std::min(ii + block_size, M); i++) {
            int i_col_A = i * lda;
            int i_col_C = i * ldc;
            for (int j = jj; j < std::min(jj + block_size, N); j++) {
              T acc = (kk == 0 ? BETA * (*C)[i_col_C + j] : (*C)[i_col_C + j]);
              for (int k = kk; k < std::min(kk + block_size, K); k++) {
                // Access B row-wise now: B[k * ldb + j]
                acc += ALPHA * (*A)[i_col_A + k] * (*B)[k * ldb + j];
              }
              (*C)[i_col_C + j] = acc;
            }
          }
        }
      }
    }
  } else {
    throw std::invalid_argument("Transposition not supported in blocked GEMM.");
  }
}

#ifdef USE_AVX_GEMM
template <typename T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         std::shared_ptr<Tensor<T>> A, int lda,
                         std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         std::shared_ptr<Tensor<T>> C, int ldc) {
  if (!A || !B || !C) {
    throw std::invalid_argument("GEMM received null tensor(s)");
  }
  if (TA == 1) A->transpose();
  if (TB == 1) B->transpose();

  // Get the pointers to the raw data
  T *a_data = A->get_raw_data().get();
  T *b_data = B->get_raw_data().get();
  T *c_data = C->get_raw_data().get();

  int i, j, k;
  int i_col, k_col, i_col_out;
  

  int simd = (256 / 8) / sizeof(T);
  int elem_left = N % simd;

  int N_s = elem_left ? N - simd : N;
  if constexpr (std::is_same<T, float>::value) {
    __m256 a_s, b_s, c_vals;
    
    __m256i mask = make_avx2_mask<T>(elem_left);
    __m256 beta_s = _mm256_broadcast_ss(&BETA);

    for (int i = 0; i < M; i++) {
      i_col = i * lda;
      i_col_out = i * ldc;

      for (j = 0; j < N_s; j += simd) {
        c_vals = _mm256_loadu_ps(c_data + i * ldc + j);
        c_vals = _mm256_mul_ps(beta_s, c_vals);

        for (int k = 0; k < K; k++) {
          k_col = k * ldb;
          float a = ALPHA * a_data[i_col + k];
          __m256 a_vals = _mm256_broadcast_ss(&a);
          __m256 b_vals = _mm256_loadu_ps(b_data + k_col + j);
          c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals);
        }
        _mm256_storeu_ps(c_data + i * ldc + j, c_vals);
      }
      if (elem_left) {
        c_vals = _mm256_loadu_ps(c_data + i_col_out + j);
        c_vals = _mm256_mul_ps(beta_s, c_vals);
        for (k = 0; k < K; k++) {
            k_col = k * ldb;
            float a = ALPHA * a_data[i_col + k];
            a_s = _mm256_broadcast_ss(&a);
            b_s = _mm256_loadu_ps(b_data + k_col + j);
            c_vals = _mm256_fmadd_ps(a_s, b_s, c_vals);
        }
        _mm256_maskstore_ps(c_data + i_col_out + j, mask, c_vals);
      }
    }
  } else if constexpr (std::is_same<T, double>::value) {
    __m256d a_s, b_s, c_vals;
    
    __m256i mask = make_avx2_mask<T>(elem_left);
    __m256d beta_s = _mm256_broadcast_sd(&BETA);

    for (int i = 0; i < M; i++) {
      i_col = i * lda;
      i_col_out = i * ldc;

      for (j = 0; j < N_s; j += simd) {
        c_vals = _mm256_loadu_pd(c_data + i * ldc + j);
        c_vals = _mm256_mul_pd(beta_s, c_vals);

        for (int k = 0; k < K; k++) {
          k_col = k * ldb;
          double a = ALPHA * a_data[i_col + k];
          __m256d a_vals = _mm256_broadcast_sd(&a);
          __m256d b_vals = _mm256_loadu_pd(b_data + k_col + j);
          c_vals = _mm256_fmadd_pd(a_vals, b_vals, c_vals);
        }
        _mm256_storeu_pd(c_data + i * ldc + j, c_vals);
      }
      if (elem_left) {
        c_vals = _mm256_loadu_pd(c_data + i_col_out + j);
        c_vals = _mm256_mul_pd(beta_s, c_vals);
        for (k = 0; k < K; k++) {
            k_col = k * ldb;
            double a = ALPHA * a_data[i_col + k];
            a_s = _mm256_broadcast_sd(&a);
            b_s = _mm256_loadu_pd(b_data + k_col + j);
            c_vals = _mm256_fmadd_pd(a_s, b_s, c_vals);
        }
        _mm256_maskstore_pd(c_data + i_col_out + j, mask, c_vals);
      }
    }
  } else if constexpr (std::is_same<T, int>::value) {
    __m256i a_s, b_s, c_vals;
    
    __m256i mask = make_avx2_mask<T>(elem_left);
    __m256i beta_s = _mm256_set1_epi32(BETA);

    for (int i = 0; i < M; i++) {
      i_col = i * lda;
      i_col_out = i * ldc;

      for (j = 0; j < N_s; j += simd) {
        c_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_data + i * ldc + j));
        c_vals = _mm256_mullo_epi32(beta_s, c_vals);

        for (int k = 0; k < K; k++) {
          k_col = k * ldb;
          int a = ALPHA * a_data[i_col + k];
          a_s = _mm256_set1_epi32(a);
          b_s = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_data + k_col + j));
          __m256i mul = _mm256_mullo_epi32(a_s, b_s);
          c_vals = _mm256_add_epi32(mul, c_vals);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(c_data + i * ldc + j), c_vals);
      }
      if (elem_left) {
        c_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c_data + i * ldc + j));
        c_vals = _mm256_mul_epi32(beta_s, c_vals);
        for (k = 0; k < K; k++) {
            k_col = k * ldb;
            int a = ALPHA * a_data[i_col + k];
            a_s = _mm256_set1_epi32(a);
            b_s = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_data + k_col + j));
            __m256i mul = _mm256_mullo_epi32(a_s, b_s);
            c_vals = _mm256_add_epi32(mul, c_vals);
        }
        _mm256_maskstore_epi32(c_data + i_col_out + j, mask, c_vals);
      }
    }
  } else {
    throw std::runtime_error("AVX2 only suppports float, double and int");
  }
  return;
}
#endif

#ifdef USE_OPENBLAS_GEMM
template <typename T>
static void mml_gemm_blas(int TA, int TB, int M, int N, int K, T ALPHA,
                          std::shared_ptr<Tensor<T>> A, int lda,
                          std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                          std::shared_ptr<Tensor<T>> C, int ldc) {
  if (TA == 1 || TB == 1) {
    throw std::invalid_argument(
        "BLAS GEMM only supports non-transposed A/B in this wrapper.");
  }

  int num_threads = std::thread::hardware_concurrency();
  openblas_set_num_threads(num_threads);

  // Build raw pointer buffers from your Tensor<T> objects
  std::vector<T> a_raw(M * K);
  std::vector<T> b_raw(K * N);
  std::vector<T> c_raw(M * N);

  // Flatten A: M x K
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      a_raw[i * K + k] = (*A)[i * lda + k];
    }
  }

  // Flatten B: K x N
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < N; ++j) {
      b_raw[k * N + j] = (*B)[k * ldb + j];
    }
  }

  // Optional: fill c_raw with values from C if BETA ≠ 0
  if (BETA != T(0)) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        c_raw[i * N + j] = (*C)[i * ldc + j];
      }
    }
  }

  if constexpr (std::is_same<T, float>::value) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA,
                a_raw.data(), K, b_raw.data(), N, BETA, c_raw.data(), N);
  } else if constexpr (std::is_same<T, double>::value) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA,
                a_raw.data(), K, b_raw.data(), N, BETA, c_raw.data(), N);
  } else {
    throw std::runtime_error("BLAS GEMM only supports float and double types.");
  }

  // Copy results back into C
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      (*C)[i * ldc + j] = c_raw[i * N + j];
    }
  }
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
static std::vector<int> mml_top_n_arg_max(
    const std::shared_ptr<const Tensor<T>> a, int n) {
  const auto size = a->get_size();
  if (size == 0) {
    throw std::runtime_error("top_n_argmax called on an empty tensor.");
  }
  if (n <= 0 || n > static_cast<int>(size)) {
    throw std::invalid_argument("Requested n is out of valid range.");
  }

  std::vector<int> indices(size);
  for (int i = 0; i < static_cast<int>(size); ++i) {
    indices[i] = i;
  }

  // Sort the values and resize it to return n number of values in order.
  std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
                    [&a](int lhs, int rhs) {
                      return (*a)[lhs] >
                             (*a)[rhs];  // Greater values come first
                    });

  indices.resize(n);
  return indices;
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
