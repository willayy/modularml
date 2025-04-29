#pragma once

#include <memory>

#include "datastructures/a_tensor.hpp"
#include "datastructures/tensor_concept.hpp"

#if defined(USE_AVX_GEMM)
template <TensorConcept::Types T>
static void mml_gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                         std::shared_ptr<Tensor<T>> A, int lda,
                         std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                         std::shared_ptr<Tensor<T>> C, int ldc);
#endif

#if defined(USE_AVX512_GEMM)
template <TensorConcept::Types T>
static void mml_gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                            std::shared_ptr<Tensor<T>> A, int lda,
                            std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                            std::shared_ptr<Tensor<T>> C, int ldc);
#endif

#include "../operations/tensor_avx_operation_functions.tpp"