#pragma once

#include "datastructures/tensor.hpp"
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

namespace TensorUtils {

/**
 * @brief Compares two tensors element-wise to check if they are close within a
 * specified tolerance.
 *
 * This std::function iterates over each element of the given tensors and checks
 * if the absolute difference between corresponding elements is within the
 * specified tolerance. If any pair of elements differ by more than the
 * tolerance, the std::function returns false. Otherwise, it returns true.
 *
 * @tparam T The data type of the tensor elements (must be an arithmetic type).
 * @param t1 The first tensor to compare.
 * @param t2 The second tensor to compare.
 * @param tolerance The maximum allowed difference between corresponding
 * elements of the tensors seen as a percentage. Default is 0.01.
 * @return true if all corresponding elements of the tensors are within the
 * specified tolerance, false otherwise.
 */
template <typename T>
bool tensors_are_close(Tensor<T> &t1, Tensor<T> &t2, T tolerance = T(0.01));

/**
 * @brief Generates a random tensor with the specified shape and value range.
 *
 * This std::function creates a tensor with random values within the specified
 * range. The type of the tensor elements must be an arithmetic type.
 *
 * @tparam T The data type of the tensor elements (must be an arithmetic type).
 * @param shape The shape of the tensor to generate.
 * @param lo_v The lower bound of the random values.
 * @param hi_v The upper bound of the random values.
 * @return A tensor with random values within the specified range.
 */
template <typename T>
Tensor<T> generate_random_tensor(const array_mml<size_t> &shape,
                                   T lo_v = T(0), T hi_v = T(1));

/**
 * @brief Initializes a tensor using the Kaiming Uniform initialization method.
 *
 * This std::function applies the Kaiming Uniform initialization to the given
 * tensor. It is commonly used for initializing weights in neural networks. This
 * version recives a seed as the final parameter.
 *
 * @tparam T The data type of the tensor elements (must be a floating-point
 * type).
 * @param W A shared pointer to the tensor to be initialized.
 * @param in_channels The number of input channels.
 * @param kernel_size The size of the kernel.
 * @param gen A random number generator.
 */
template <typename T>
void kaiming_uniform(std::shared_ptr<Tensor<T>> W, size_t in_channels,
                     size_t kernel_size, std::mt19937 &gen);

/**
 * @brief Initializes a tensor using the Kaiming Uniform initialization method.
 *
 * This std::function applies the Kaiming Uniform initialization to the given
 * tensor. It is commonly used for initializing weights in neural networks. This
 * version of the std::function uses an internal random number generator.
 *
 * @tparam T The data type of the tensor elements (must be a floating-point
 * type).
 * @param W A shared pointer to the tensor to be initialized.
 * @param in_channels The number of input channels.
 * @param kernel_size The size of the kernel.
 */
template <typename T>
void kaiming_uniform(std::shared_ptr<Tensor<T>> W, size_t in_channels,
                     size_t kernel_size);
               
}

#define _TENSOR_UTILS(DT) \
template bool TensorUtils::tensors_are_close<DT>(Tensor<DT>&, Tensor<DT>&, DT); \
template Tensor<DT> TensorUtils::generate_random_tensor<DT>(const array_mml<size_t>&, DT, DT);

#define _TENSOR_UTILS_REAL(DT) \
template void TensorUtils::kaiming_uniform<DT>(std::shared_ptr<Tensor<DT>>, size_t, size_t, std::mt19937&); \
template void TensorUtils::kaiming_uniform<DT>(std::shared_ptr<Tensor<DT>>, size_t, size_t);
