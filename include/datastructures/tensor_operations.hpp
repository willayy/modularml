#pragma once

#include <memory>

#include "datastructures/tensor.hpp"

template <typename T>
class TensorOperations {
 public:
  TensorOperations() = delete;  // Prevent instantiation of this class

  static void gemm(int TA, int TB, int M, int N, int K, T ALPHA, T BETA,
                   std::shared_ptr<Tensor<T>> A, int lda,
                   std::shared_ptr<Tensor<T>> B, int ldb,
                   std::shared_ptr<Tensor<T>> C, int ldc);

  static void add(const std::shared_ptr<const Tensor<T>> a,
                  const std::shared_ptr<const Tensor<T>> b,
                  std::shared_ptr<Tensor<T>> c);

  static void subtract(const std::shared_ptr<Tensor<T>> a,
                       const std::shared_ptr<Tensor<T>> b,
                       std::shared_ptr<Tensor<T>> c);

  static void multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                       std::shared_ptr<Tensor<T>> c);

  static bool equals(const std::shared_ptr<Tensor<T>> a,
                     const std::shared_ptr<Tensor<T>> b);

  static int arg_max(const std::shared_ptr<const Tensor<T>> a);

  static void elementwise(const std::shared_ptr<const Tensor<T>> a,
                          const std::function<T(T)> &f,
                          const std::shared_ptr<Tensor<T>> c);

  static void elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                   const std::function<T(T)> &f);

  static void sliding_window(
      const array_mml<size_t> &in_shape, const array_mml<size_t> &out_shape,
      const std::vector<int> &kernel_shape, const std::vector<int> &strides,
      const std::vector<int> &dilations,
      const std::vector<std::pair<int, int>> &pads,
      const std::function<void(const std::vector<std::vector<size_t>> &,
                               const std::vector<size_t> &)> &window_f);
};

#define _TENSOR_OPERATIONS(DT) template class TensorOperations<DT>;