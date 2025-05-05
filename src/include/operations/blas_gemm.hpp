#pragma once

#include <memory>

#include "datastructures/a_tensor.hpp"
#include "datastructures/tensor_concept.hpp"

#ifdef USE_OPENBLAS_GEMM
template <TensorConcept::Types T>
static void mml_gemm_blas(int TA, int TB, int M, int N, int K, T ALPHA,
                          std::shared_ptr<Tensor<T>> A, int lda,
                          std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                          std::shared_ptr<Tensor<T>> C, int ldc);
#include "../datastructures/blas_gemm.tpp"
#endif
