#include "datastructures/tensor_concept.hpp"
#include "operations/blas_gemm.hpp"

#ifdef USE_OPENBLAS_GEMM
template <TensorConcept::Types T>
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