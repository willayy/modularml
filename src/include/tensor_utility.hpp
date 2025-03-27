#pragma once

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"

/**
 * @brief Compares two tensors element-wise to check if they are close within a
 * specified tolerance.
 *
 * This function iterates over each element of the given tensors and checks if
 * the absolute difference between corresponding elements is within the
 * specified tolerance. If any pair of elements differ by more than the
 * tolerance, the function returns false. Otherwise, it returns true.
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

#define GENERATE_RANDOM_ARRAY_INTEGRAL(T)            \
  (std::is_integral_v<T>,                            \
   "Random array generation (integral) requires an " \
   "integral type (int, long, etc.).");
template <typename T>
array_mml<T> generate_random_array_mml_integral(uli lo_sz = 1, uli hi_sz = 5,
                                                T lo_v = 1, T hi_v = 10);

#define GENERATE_RANDOM_ARRAY_REAL(T)           \
  (std::is_floating_point_v<T>,                 \
   "Random array generation (real) requires a " \
   "floating-point type (float, double, etc.).");
template <typename T>
array_mml<T> generate_random_array_mml_real(uli lo_sz = 1, uli hi_sz = 5,
                                            T lo_v = 1, T hi_v = 100);

/**
 * @brief Generates a random tensor with the specified shape and value range.
 *
 * This function creates a tensor with random values within the specified range.
 * The type of the tensor elements must be an arithmetic type.
 *
 * @tparam T The data type of the tensor elements (must be an arithmetic type).
 * @param shape The shape of the tensor to generate.
 * @param lo_v The lower bound of the random values.
 * @param hi_v The upper bound of the random values.
 * @return A tensor with random values within the specified range.
 */
#define GENERATE_RANDOM_TENSOR(T)                     \
  (std::is_arithmetic_v<T>,                           \
   "Random Tensor generation requires an arithmetic " \
   "type (int, float, double, etc.).");
template <typename T>
static auto generate_random_tensor(const array_mml<uli> &shape, T lo_v = T(0), T hi_v = T(1));

/**
 * @brief Initializes a tensor using the Kaiming Uniform initialization method.
 *
 * This function applies the Kaiming Uniform initialization to the given tensor.
 * It is commonly used for initializing weights in neural networks.
 * This version recives a seed as the final parameter.
 *
 * @tparam T The data type of the tensor elements (must be a floating-point type).
 * @param W A shared pointer to the tensor to be initialized.
 * @param in_channels The number of input channels.
 * @param kernel_size The size of the kernel.
 * @param gen A random number generator.
 */
template <typename T>
void kaimingUniform(shared_ptr<Tensor<T>> W, uli in_channels, uli kernel_size, std::mt19937 &gen);

/**
 * @brief Initializes a tensor using the Kaiming Uniform initialization method.
 *
 * This function applies the Kaiming Uniform initialization to the given tensor.
 * It is commonly used for initializing weights in neural networks.
 * This version of the function uses an internal random number generator.
 *
 * @tparam T The data type of the tensor elements (must be a floating-point type).
 * @param W A shared pointer to the tensor to be initialized.
 * @param in_channels The number of input channels.
 * @param kernel_size The size of the kernel.
 */
template <typename T>
void kaimingUniform(shared_ptr<Tensor<T>> W, uli in_channels, uli kernel_size);

#include "../tensor_utility.tpp"