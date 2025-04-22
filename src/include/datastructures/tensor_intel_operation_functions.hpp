#pragma once
#include <memory>

#include "datastructures/a_tensor.hpp"

template <typename T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               std::shared_ptr<Tensor<T>> C, int ldc);

#include "../datastructures/tensor_intel_operation_functions.tpp"