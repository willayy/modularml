#include <immintrin.h>

#include "datastructures/tensor_operations.hpp"

template <typename T>
void TensorOperations<T>::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                               T BETA, std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  if (TA == 1) A = A->transpose();
  if (TB == 1) B = B->transpose();
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

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE