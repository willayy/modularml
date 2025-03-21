#pragma once

#include "a_tensor.hpp"
#include "globals.hpp"

#define ASSERT_ALLOWED_TYPES_AR(T) static_assert(std::is_arithmetic_v<T>, "ArithmeticModule type must be arithmetic.")

/// @brief A module for performing simple arithmetic operations on tensor structures.
/// @tparam T the data type (numeric).
template <typename T>
class ArithmeticModule {
 public:
  /// @brief Default constructor for ArithmeticModule class.
  ArithmeticModule() = default;

  /// @brief Copy constructor for ArithmeticModule class.
  ArithmeticModule(const ArithmeticModule &other) = default;

  /// @brief Move constructor for ArithmeticModule class.
  ArithmeticModule(ArithmeticModule &&other) noexcept = default;

  /// @brief Abstract destructor for ArithmeticModule class.
  virtual ~ArithmeticModule() = default;

  /// @brief Add two tensor structures.
  /// @param a The first tensor.
  /// @param b The second tensor structure.
  /// @param c The tensor structure to store the result.
  virtual void add(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Subtract one tensor structure from another.
  /// @param a The tensor structure to subtract from.
  /// @param b The tensor structure to subtract.
  /// @param c The tensor structure to store the result.
  virtual void subtract(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Multiply a tensor structure by a scalar.
  /// @param a The tensor structure.
  /// @param b The scalar value.
  /// @param c The tensor structure to store the result.
  virtual void multiply(const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Check if two tensor structures are equal.
  /// @param a The first tensor structure.
  /// @param b The second tensor structure.
  virtual bool equals(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b) const = 0;

  /// @brief Apply an element-wise operation to two tensor structures.
  /// @param a The tensor structure.
  /// @param f The function to apply element-wise.
  /// @param c The tensor structure to store the result.
  virtual void elementwise(const shared_ptr<const Tensor<T>> a, T (*f)(T), const shared_ptr<Tensor<T>> c) const = 0;

  /// @brief Apply an element-wise operation to a tensor structure in place.
  /// @param a The tensor structure.
  /// @param f The function to apply element-wise.
  virtual void elementwise_in_place(const shared_ptr<Tensor<T>> a, T (*f)(T)) const = 0;

};