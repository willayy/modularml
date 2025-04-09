#pragma once

#include "datastructures/a_tensor.hpp"
#include "../utility/uli.hpp"
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

#define ASSERT_ALLOWED_TYPES_AR(T)                                             \
  static_assert(std::is_arithmetic_v<T>,                                       \
                "ArithmeticModule type must be arithmetic.")

/// @brief A module for performing simple arithmetic operations on tensor
/// structures.
/// @tparam T the data type (numeric).
template <typename T> class ArithmeticModule {
public:
  /// @brief Default constructor for ArithmeticModule class.
  [[deprecated("Use TensorOperationsModule instead")]]
  ArithmeticModule() = default;

  /// @brief Copy constructor for ArithmeticModule class.
  [[deprecated("Use TensorOperationsModule instead")]]
  ArithmeticModule(const ArithmeticModule &other) = default;

  /// @brief Move constructor for ArithmeticModule class.
  [[deprecated("Use TensorOperationsModule instead")]]
  ArithmeticModule(ArithmeticModule &&other) noexcept = default;

  /// @brief Abstract destructor for ArithmeticModule class.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual ~ArithmeticModule() = default;

  /// @brief Add two tensor structures.
  /// @param a The first tensor.
  /// @param b The second tensor structure.
  /// @param c The tensor structure to store the result.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual void add(const std::shared_ptr<Tensor<T>> a,
                   const std::shared_ptr<Tensor<T>> b,
                   std::shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Subtract one tensor structure from another.
  /// @param a The tensor structure to subtract from.
  /// @param b The tensor structure to subtract.
  /// @param c The tensor structure to store the result.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual void subtract(const std::shared_ptr<Tensor<T>> a,
                        const std::shared_ptr<Tensor<T>> b,
                        std::shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Multiply a tensor structure by a scalar.
  /// @param a The tensor structure.
  /// @param b The scalar value.
  /// @param c The tensor structure to store the result.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual void multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                        std::shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Check if two tensor structures are std::equal.
  /// @param a The first tensor structure.
  /// @param b The second tensor structure.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual bool equals(const std::shared_ptr<Tensor<T>> a,
                      const std::shared_ptr<Tensor<T>> b) const = 0;

  /// @brief Returns the index of the maximum value in a tensor (flattened).
  ///
  /// A simplified version of argMax that mimics PyTorch behavior:
  /// it scans the tensor in row-major order and returns the **first index**
  /// where the maximum value appears. No axis support, no tie-breaking control.
  ///
  /// This is much simpler than the ONNX ArgMax operator, which supports
  /// multi-axis reductions, dimension retention, and tie-breaking options.
  ///
  /// @param a The input tensor to search through.
  /// @return The flattened index (int) of the first occurrence of the maximum
  /// value.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual int arg_max(const std::shared_ptr<const Tensor<T>> a) const = 0;

  /// @brief Apply an element-wise operation to two tensor structures.
  /// @param a The tensor structure.
  /// @param f The std::function to apply element-wise.
  /// @param c The tensor structure to store the result.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual void elementwise(const std::shared_ptr<const Tensor<T>> a,
                           std::function<T(T)> f,
                           const std::shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Apply an element-wise operation to a tensor structure in place.
  /// @param a The tensor structure.
  /// @param f The std::function to apply element-wise.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual void elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                    std::function<T(T)> f) const = 0;
};