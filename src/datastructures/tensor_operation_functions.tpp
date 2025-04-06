#pragma once
#include "datastructures/tensor_operation_functions.hpp"

#if defined(USE_AVX_GEMM)
#include <immintrin.h>
#endif

template <typename T>
static void mml_gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   std::shared_ptr<Tensor<T>> A, int lda,
                                   std::shared_ptr<Tensor<T>> B, int ldb,
                                   T BETA, std::shared_ptr<Tensor<T>> C,
                                   int ldc) {
  int k_col;
  int i_col_out;

  if (TA == 1)
    std::invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    std::invalid_argument("Transpose B not yet supported.");

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

  if (TA == 1)
    std::invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    std::invalid_argument("Transpose B not yet supported.");

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

  if (TA == 1)
    std::invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    std::invalid_argument("Transpose B not yet supported.");

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

  if (TA == 1)
    std::invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    std::invalid_argument("Transpose B not yet supported.");

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
                             shared_ptr<Tensor<T>> A, int lda,
                             shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             shared_ptr<Tensor<T>> C, int ldc) {
  
  int block_size = 128; // This depends on the CPU architecture - We can look into having the size of this be dynamically fetched
  for (int jj = 0; jj < N; jj += block_size) {
    for (int kk = 0; kk < K; kk += block_size) {
      for (int ii = 0; ii < M; ii += block_size) {
        
        for (int j = jj; j < std::min(jj + block_size, N); j++) {
          for (int i = ii; i < std::min(ii + block_size, M); i++) {
            T sum = 0;
            for (int k = kk; k < std::min(k + block_size, K); k++) {
              sum += (*A)[i * lda + k] * (*B)[k * ldb + j];
            }
            (*C)[i * ldc + j] += ALPHA * sum + BETA * (*C)[i * ldc + j];
          }  
        }
      }
    }
  }
}

template <typename T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         shared_ptr<Tensor<T>> A, int lda,
                         shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         shared_ptr<Tensor<T>> C, int ldc) {
  
}

template <typename T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            std::shared_ptr<Tensor<T>> A, int lda,
                            std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("AVX-512 GEMM not yet supported.");
}

template <typename T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("Intel MKL GEMM not yet supported.");
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_inner_product(std::shared_ptr<Tensor<T>> A,
                            std::shared_ptr<Tensor<T>> B, float alpha,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_inner_product(transA, transB, M, N, K, static_cast<T>(alpha), A, lda,
                         B, ldb, static_cast<T>(beta), C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_outer_product(std::shared_ptr<Tensor<T>> A,
                            std::shared_ptr<Tensor<T>> B, float alpha,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_outer_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta,
                         C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_row_wise_product(std::shared_ptr<Tensor<T>> A,
                               std::shared_ptr<Tensor<T>> B, float alpha,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_row_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_col_wise_product(std::shared_ptr<Tensor<T>> A,
                               std::shared_ptr<Tensor<T>> B, float alpha,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_col_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_blocked(std::shared_ptr<Tensor<T>> A,
                      std::shared_ptr<Tensor<T>> B, float alpha, float beta,
                      int transA, int transB,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_blocked(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                   ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_avx(std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B,
                  float alpha, float beta, int transA, int transB,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_avx(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_avx512(std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B,
                     float alpha, float beta, int transA, int transB,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_avx512(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                  ldc);
  return C_p;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_onnx_gemm_intel_MKL(std::shared_ptr<Tensor<T>> A,
                        std::shared_ptr<Tensor<T>> B, float alpha, float beta,
                        int transA, int transB,
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
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
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
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
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
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
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