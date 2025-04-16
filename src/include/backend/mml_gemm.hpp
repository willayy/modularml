#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "datastructures/tensor.hpp"

namespace Gemm {
  template <typename T>
  void inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                          std::shared_ptr<Tensor<T>> A, int lda,
                          std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                          std::shared_ptr<Tensor<T>> C, int ldc);
  
  template <typename T>
  void outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                          std::shared_ptr<Tensor<T>> A, int lda,
                          std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                          std::shared_ptr<Tensor<T>> C, int ldc);

  template <typename T>
  void row_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             std::shared_ptr<Tensor<T>> A, int lda,
                             std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             std::shared_ptr<Tensor<T>> C, int ldc);

  template <typename T>
  void col_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                             std::shared_ptr<Tensor<T>> A, int lda,
                             std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                             std::shared_ptr<Tensor<T>> C, int ldc);
};

#define _GEMM(DT) \
template void Gemm::inner_product<DT>(int, int, int, int, int, DT, std::shared_ptr<Tensor<DT>>, int, std::shared_ptr<Tensor<DT>>, int, DT, std::shared_ptr<Tensor<DT>>, int); \
template void Gemm::outer_product<DT>(int, int, int, int, int, DT, std::shared_ptr<Tensor<DT>>, int, std::shared_ptr<Tensor<DT>>, int, DT, std::shared_ptr<Tensor<DT>>, int); \
template void Gemm::row_wise_product<DT>(int, int, int, int, int, DT, std::shared_ptr<Tensor<DT>>, int, std::shared_ptr<Tensor<DT>>, int, DT, std::shared_ptr<Tensor<DT>>, int); \
template void Gemm::col_wise_product<DT>(int, int, int, int, int, DT, std::shared_ptr<Tensor<DT>>, int, std::shared_ptr<Tensor<DT>>, int, DT, std::shared_ptr<Tensor<DT>>, int);
