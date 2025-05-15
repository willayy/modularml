#include <immintrin.h>

#include "datastructures/tensor_operations.hpp"
#include "utility/avx_mask_helper.hpp"

template <typename T>
void TensorOperations<T>::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                               T BETA, std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  if (!A || !B || !C) {
    throw std::invalid_argument("GEMM received null tensor(s)");
  }
  if (TA == 1) A = A->transpose();
  if (TB == 1) B = B->transpose();

  // Validate
  auto a_shape = A->get_shape();
  auto b_shape = B->get_shape();
  auto c_shape = C->get_shape();

  if (a_shape[0] != M || a_shape[1] != K)
    throw std::invalid_argument("Matrix A shape does not match M x K");

  if (b_shape[0] != K || b_shape[1] != N)
    throw std::invalid_argument("Matrix B shape does not match K x N");

  if (c_shape[0] != M || c_shape[1] != N)
    throw std::invalid_argument("Matrix C shape does not match M x N");

  // Get the pointers to the raw data
  T *a_data = A->get_raw_data().get();
  T *b_data = B->get_raw_data().get();
  T *c_data = C->get_raw_data().get();

  int i, j, k;
  int i_col, k_col, i_col_out;

  int simd = (512 / 8) / sizeof(T);
  int elem_left = N % simd;

  int N_s = elem_left ? N - simd : N;
  if constexpr (std::is_same<T, float>::value) {
    __m512 a_s, b_s, c_vals;

    auto mask = make_avx512_mask<T>(elem_left);
    __m512 beta_s = _mm512_set1_ps(BETA);

    for (int i = 0; i < M; i++) {
      i_col = i * lda;
      i_col_out = i * ldc;

      for (j = 0; j < N_s; j += simd) {
        c_vals = _mm512_loadu_ps(c_data + i * ldc + j);
        c_vals = _mm512_mul_ps(beta_s, c_vals);

        for (int k = 0; k < K; k++) {
          k_col = k * ldb;
          float a = ALPHA * a_data[i_col + k];
          __m512 a_vals = _mm512_set1_ps(a);
          __m512 b_vals = _mm512_loadu_ps(b_data + k_col + j);
          c_vals = _mm512_fmadd_ps(a_vals, b_vals, c_vals);
        }
        _mm512_storeu_ps(c_data + i * ldc + j, c_vals);
      }
      if (elem_left) {
        c_vals = _mm512_loadu_ps(c_data + i_col_out + j);
        c_vals = _mm512_mul_ps(beta_s, c_vals);
        for (k = 0; k < K; k++) {
          k_col = k * ldb;
          float a = ALPHA * a_data[i_col + k];
          a_s = _mm512_set1_ps(a);
          b_s = _mm512_loadu_ps(b_data + k_col + j);
          c_vals = _mm512_fmadd_ps(a_s, b_s, c_vals);
        }
        _mm512_mask_storeu_ps(c_data + i_col_out + j, mask, c_vals);
      }
    }
  } else if constexpr (std::is_same<T, double>::value) {
    __m512d a_s, b_s, c_vals;

    auto mask = make_avx512_mask<T>(elem_left);
    __m512d beta_s = _mm512_set1_pd(BETA);

    for (int i = 0; i < M; i++) {
      i_col = i * lda;
      i_col_out = i * ldc;

      for (j = 0; j < N_s; j += simd) {
        c_vals = _mm512_loadu_pd(c_data + i * ldc + j);
        c_vals = _mm512_mul_pd(beta_s, c_vals);

        for (int k = 0; k < K; k++) {
          k_col = k * ldb;
          double a = ALPHA * a_data[i_col + k];
          __m512d a_vals = _mm512_set1_pd(a);
          __m512d b_vals = _mm512_loadu_pd(b_data + k_col + j);
          c_vals = _mm512_fmadd_pd(a_vals, b_vals, c_vals);
        }
        _mm512_storeu_pd(c_data + i * ldc + j, c_vals);
      }
      if (elem_left) {
        c_vals = _mm512_loadu_pd(c_data + i_col_out + j);
        c_vals = _mm512_mul_pd(beta_s, c_vals);
        for (k = 0; k < K; k++) {
          k_col = k * ldb;
          double a = ALPHA * a_data[i_col + k];
          a_s = _mm512_set1_pd(a);
          b_s = _mm512_loadu_pd(b_data + k_col + j);
          c_vals = _mm512_fmadd_pd(a_s, b_s, c_vals);
        }
        _mm512_mask_storeu_pd(c_data + i_col_out + j, mask, c_vals);
      }
    }
  } else if constexpr (std::is_same<T, int>::value) {
    __m512i a_s, b_s, c_vals;

    auto mask = make_avx512_mask<T>(elem_left);
    __m512i beta_s = _mm512_set1_epi32(BETA);

    for (int i = 0; i < M; i++) {
      i_col = i * lda;
      i_col_out = i * ldc;

      for (j = 0; j < N_s; j += simd) {
        c_vals = _mm512_loadu_si512(
            reinterpret_cast<const __m512i *>(c_data + i * ldc + j));
        c_vals = _mm512_mullo_epi32(beta_s, c_vals);

        for (int k = 0; k < K; k++) {
          k_col = k * ldb;
          int a = ALPHA * a_data[i_col + k];
          a_s = _mm512_set1_epi32(a);
          b_s = _mm512_loadu_si512(
              reinterpret_cast<const __m512i *>(b_data + k_col + j));
          __m512i mul = _mm512_mullo_epi32(a_s, b_s);
          c_vals = _mm512_add_epi32(mul, c_vals);
        }
        _mm512_storeu_si512(reinterpret_cast<__m512i *>(c_data + i * ldc + j),
                            c_vals);
      }
      if (elem_left) {
        c_vals = _mm512_loadu_si512(
            reinterpret_cast<const __m512i *>(c_data + i * ldc + j));
        c_vals = _mm512_mul_epi32(beta_s, c_vals);
        for (k = 0; k < K; k++) {
          k_col = k * ldb;
          int a = ALPHA * a_data[i_col + k];
          a_s = _mm512_set1_epi32(a);
          b_s = _mm512_loadu_si512(
              reinterpret_cast<const __m512i *>(b_data + k_col + j));
          __m512i mul = _mm512_mullo_epi32(a_s, b_s);
          c_vals = _mm512_add_epi32(mul, c_vals);
        }
        _mm512_mask_storeu_epi32(c_data + i_col_out + j, mask, c_vals);
      }
    }
  } else {
    throw std::runtime_error("AVX512 only suppports float, double and int");
  }
  return;
}

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE