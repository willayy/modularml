#pragma once
#include "datastructures/tensor_operation_functions.hpp"
#include "datastructures/mml_tensor.hpp"

#include <immintrin.h>

template <typename T>
static void mml_gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   shared_ptr<Tensor<T>> A, int lda,
                                   shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                   shared_ptr<Tensor<T>> C, int ldc) {
  int k_col;
  int i_col_out;

  if (TA == 1)
    throw invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    throw invalid_argument("Transpose B not yet supported.");

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
                                   shared_ptr<Tensor<T>> A, int lda,
                                   shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                   shared_ptr<Tensor<T>> C, int ldc) {
  int i_col;
  int k_col;
  int i_col_out;

  if (TA == 1)
    throw invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    throw invalid_argument("Transpose B not yet supported.");

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
                                      T ALPHA, shared_ptr<Tensor<T>> A, int lda,
                                      shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                      shared_ptr<Tensor<T>> C, int ldc) {

  int i_col;
  int k_col;
  int i_col_out;

  if (TA == 1)
    throw invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    throw invalid_argument("Transpose B not yet supported.");

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
                                      T ALPHA, shared_ptr<Tensor<T>> A, int lda,
                                      shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                      shared_ptr<Tensor<T>> C, int ldc) {
  int i_col;
  int k_col;
  int i_col_out;

  if (TA == 1)
    throw invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    throw invalid_argument("Transpose B not yet supported.");

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
  
  int block_size = 64; // This depends on the CPU architecture - We can look into having the size of this be dynamically fetched
  int k_col;
  int i_col_out;

  if (TA == 1)
    throw invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    throw invalid_argument("Transpose B not yet supported.");

  for (int i_block = 0; i_block < M; i_block += block_size) {
    int i_end = std::min(i_block + block_size, M);

    for (int j_block = 0; j_block < N; j_block += block_size) {
      int j_end = std::min(j_block + block_size, N);

      for (int i = i_block; i < i_end; i++) {
        i_col_out = i * ldc;
        for (int j = j_block; j < j_end; j++) {
          (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
        }
      }

      for (int k_block = 0; k_block < K; k_block += block_size) {
        int k_end = std::min(k_block + block_size, K);

        for (int i = i_block; i < i_end; i++) {
          i_col_out = i * ldc;
          for (int k = k_block; k < k_end; k++) {
            k_col = k * ldb;
            for (int j = j_block; j < j_end; j++) {
              (*C)[i_col_out + j] += ((T)ALPHA) * (*A)[i * lda + k] * (*B)[k_col + j];
            }
          }
        }
      }
    }
  }

  return;
}
  

template <typename T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         shared_ptr<Tensor<T>> A, int lda,
                         shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         shared_ptr<Tensor<T>> C, int ldc) {
  if (TA == 1)
    throw invalid_argument("Transpose A not yet supported for AVX2 GEMM.");
  if (TB == 1)
    throw invalid_argument("Transpose B not yet supported for AVX2 GEMM.");
  
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
  }
  else if constexpr (std::is_same<T, double>::value) {
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
  }
  else if constexpr (std::is_same<T, int>::value) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 8) {

        __m256i sum = _mm256_setzero_si256();
  
        for (int k = 0; k < K; k++) {
  
          int a_scalar = (*A)[i * lda + k]; 
          __m256i a_broadcast = _mm256_set1_epi32(a_scalar);  

          __m256i b_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&(*B)[k * ldb + j]));
          __m256i product = _mm256_mullo_epi32(a_broadcast, b_vals);

          sum = _mm256_add_epi32(sum, product);
        }
    
        sum = _mm256_mullo_epi32(sum, _mm256_set1_epi32(ALPHA));
        sum = _mm256_add_epi32(sum, _mm256_set1_epi32(BETA));
    
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&(*C)[i * ldc + j]), sum);
      }
    }
  }
  else {
    throw runtime_error("AVX2 only suppports double, float or int");
  }
  return;
}

template <typename T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            shared_ptr<Tensor<T>> A, int lda,
                            shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            shared_ptr<Tensor<T>> C, int ldc) {
  if (TA == 1)
    invalid_argument("Transpose A not yet supported for AVX-512 GEMM.");
  if (TB == 1)
    invalid_argument("Transpose B not yet supported for AVX-512 GEMM.");
  
  
  if constexpr(std::is_same<T, float>::value) {
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
  } 
  else if constexpr(std::is_same<T, double>::value) {
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
  } 
  else if constexpr(std::is_same<T, int>::value) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 16) {
        __m512i sum = _mm512_setzero_si512();

        for (int k = 0; k < K; ++k) {
          __m512i a_vals = _mm512_set1_epi32((*A)[i * lda + k]); // scalar broadcast

          __m512i b_vals = _mm512_loadu_si512(reinterpret_cast<const void*>(&(*B)[k * ldb + j]));

          __m512i product = _mm512_mullo_epi32(a_vals, b_vals);
          sum = _mm512_add_epi32(sum, product);
        }

        sum = _mm512_mullo_epi32(_mm512_set1_epi32(ALPHA), sum);
        sum = _mm512_add_epi32(sum, _mm512_set1_epi32(BETA));

        _mm512_storeu_si512(reinterpret_cast<void*>(&(*C)[i * ldc + j]), sum);
      }
    }
  }
  else {
    throw runtime_error("AVX-512 only suppports double, float or int");
  }
}

template <typename T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               shared_ptr<Tensor<T>> A, int lda,
                               shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               shared_ptr<Tensor<T>> C, int ldc) {
  invalid_argument("Intel MKL GEMM not yet supported.");
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_inner_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                            float alpha, float beta, int transA, int transB,
                            optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_inner_product(transA, transB, M, N, K, static_cast<T>(alpha), A, lda,
                         B, ldb, static_cast<T>(beta), C_p, ldc);
  return C_p;
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_outer_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                            float alpha, float beta, int transA, int transB,
                            optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_outer_product(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta,
                         C_p, ldc);
  return C_p;
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_row_wise_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                               float alpha, float beta, int transA, int transB,
                               optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_row_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_col_wise_product(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                               float alpha, float beta, int transA, int transB,
                               optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_col_wise_product(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                            beta, C_p, ldc);
  return C_p;
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_blocked(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                      float alpha, float beta, int transA, int transB,
                      optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_blocked(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                   ldc);
  return C_p;
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_avx(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha,
                  float beta, int transA, int transB,
                  optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_avx(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p, ldc);
  return C_p;
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_avx512(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                     float alpha, float beta, int transA, int transB,
                     optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_avx512(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                  ldc);
  return C_p;
}

template <typename T>
static shared_ptr<Tensor<T>>
mml_onnx_gemm_intel_MKL(shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                        float alpha, float beta, int transA, int transB,
                        optional<shared_ptr<Tensor<T>>> C) {
  const auto shape_A = A->get_shape();
  const auto shape_B = B->get_shape();
  const int M = (int)shape_A[0];
  const int N = (int)shape_B[1];
  const int K = (int)shape_A[1];
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  shared_ptr<Tensor<T>> C_p =
      C.has_value() ? *C : TensorFactory::create_tensor<T>({M, N});
  mml_gemm_intel(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_p,
                 ldc);
  return C_p;
}

template <typename T>
static void mml_add(const shared_ptr<const Tensor<T>> a,
                    const shared_ptr<const Tensor<T>> b,
                    shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (uli i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
static void mml_subtract(const shared_ptr<Tensor<T>> a,
                         const shared_ptr<Tensor<T>> b,
                         shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (uli i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
static void mml_multiply(const shared_ptr<Tensor<T>> a, const T b,
                         shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (uli i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
static bool mml_equals(const shared_ptr<Tensor<T>> a,
                       const shared_ptr<Tensor<T>> b) {
  if (a->get_size() != b->get_size() || a->get_shape() != b->get_shape()) {
    return false;
  } else {
    const auto size = a->get_size();
    for (uli i = 0; i < size; i++) {
      if ((*a)[i] != (*b)[i]) {
        return false;
      }
    }
    return true;
  }
}

template <typename T>
static void mml_elementwise(const shared_ptr<const Tensor<T>> a,
                            const function<T(T)> &f,
                            const shared_ptr<Tensor<T>> c) {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<uli> indices(num_dimensions);
  for (uli i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = a->get_size();

  for (uli linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply function `f` from `a` to `c`
    (*c)[indices] = f((*a)[indices]);

    // Increment indices
    uli d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    } while (d-- > 0);
  }
}

template <typename T>
static void mml_elementwise_in_place(const shared_ptr<Tensor<T>> a,
                                     const function<T(T)> &f) {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<uli> indices(num_dimensions);
  for (uli i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }

  const auto total_elements = a->get_size();

  for (uli linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply the function `f` to the current element
    (*a)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    uli d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    } while (d-- > 0);
  }
}

template <typename T>
static int mml_arg_max(const shared_ptr<const Tensor<T>> a) {
  const auto size = a->get_size();
  if (size == 0) {
    throw runtime_error("arg_max called on an empty tensor.");
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