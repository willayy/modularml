#pragma once

#include "datastructures/tensor_operation_functions.hpp"

template <typename T>
static void mml_gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                   shared_ptr<Tensor<T>> A, int lda,
                                   shared_ptr<Tensor<T>> B, int ldb, T BETA,
                                   shared_ptr<Tensor<T>> C, int ldc) {
  int k_col;
  int i_col_out;

  if (TA == 1)
    invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    invalid_argument("Transpose B not yet supported.");

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
    invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    invalid_argument("Transpose B not yet supported.");

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
    invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    invalid_argument("Transpose B not yet supported.");

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
    invalid_argument("Transpose A not yet supported.");
  if (TB == 1)
    invalid_argument("Transpose B not yet supported.");

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
  invalid_argument("Blocked GEMM not yet supported.");
}

template <typename T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         shared_ptr<Tensor<T>> A, int lda,
                         shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         shared_ptr<Tensor<T>> C, int ldc) {
  invalid_argument("AVX GEMM not yet supported.");
}

template <typename T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            shared_ptr<Tensor<T>> A, int lda,
                            shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            shared_ptr<Tensor<T>> C, int ldc) {
  invalid_argument("AVX-512 GEMM not yet supported.");
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
      C.has_value()
          ? *C
          : TensorFactory::create_tensor({M, N});
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
      C.has_value() ? *C
                    : TensorFactory::create_tensor({M, N});
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
      C.has_value() ? *C
                    : TensorFactory::create_tensor({M, N});
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
      C.has_value() ? *C
                    : TensorFactory::create_tensor({M, N});
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
      C.has_value() ? *C
                    : TensorFactory::create_tensor({M, N});
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
      C.has_value() ? *C
                    : TensorFactory::create_tensor({M, N});
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
      C.has_value() ? *C
                    : TensorFactory::create_tensor({M, N});
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
      C.has_value() ? *C
                    : TensorFactory::create_tensor({M, N});
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