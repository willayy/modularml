#pragma once

#include <functional>
#include <memory>

#include "datastructures/a_tensor.hpp"

/// @brief Tensor Operations Function Types
namespace toft {
template <typename T>
using gemm_func = std::function<void(
    int TA, int TB, int M, int N, int K, T ALPHA,
    std::shared_ptr<Tensor<T>> A, int lda, std::shared_ptr<Tensor<T>> B,
    int ldb, T BETA, std::shared_ptr<Tensor<T>> C, int ldc)>;

template <typename T>
using gemm_onnx_func = std::function<std::shared_ptr<Tensor<T>>(
    std::shared_ptr<Tensor<T>> A, std::shared_ptr<Tensor<T>> B,
    float alpha, float beta, int transA, int transB,
    std::optional<std::shared_ptr<Tensor<T>>> C)>;

template <typename T>
using add_func =
    std::function<void(const std::shared_ptr<const Tensor<T>> a,
                       const std::shared_ptr<const Tensor<T>> b,
                       std::shared_ptr<Tensor<T>> c)>;

template <typename T>
using subtract_func = std::function<void(const std::shared_ptr<Tensor<T>> a,
                                         const std::shared_ptr<Tensor<T>> b,
                                         std::shared_ptr<Tensor<T>> c)>;

template <typename T>
using multiply_func =
    std::function<void(const std::shared_ptr<Tensor<T>> a, const T b,
                       std::shared_ptr<Tensor<T>> c)>;

template <typename T>
using equals_func = std::function<bool(const std::shared_ptr<Tensor<T>> a,
                                       const std::shared_ptr<Tensor<T>> b)>;

template <typename T>
using elementwise_func =
    std::function<void(const std::shared_ptr<const Tensor<T>> a,
                       const std::function<T(T)>& f,
                       const std::shared_ptr<Tensor<T>> c)>;

template <typename T>
using elementwise_in_place_func =
    std::function<void(const std::shared_ptr<Tensor<T>> a,
                       const std::function<T(T)>& f)>;

template <typename T>
using arg_max_func =
    std::function<int(const std::shared_ptr<const Tensor<T>> a)>;


template <typename T>
using sliding_window_func = std::function<void(
    const array_mml<size_t>& in_shape, const array_mml<size_t>& out_shape,
    const std::vector<int>& kernel_shape, const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const std::vector<std::pair<int, int>>& pads,
    const std::function<void(const std::vector<std::vector<size_t>>&,
                             const std::vector<size_t>&)>& window_f)>;

}  // namespace tfd