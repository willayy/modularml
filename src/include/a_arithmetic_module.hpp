#pragma once

#include <type_traits>

#include "a_data_structure.hpp"

#define ASSERT_ALLOWED_TYPES(T) static_assert(std::is_arithmetic_v<T>, "Data structure type must be an arithmetic type.")

/// @brief A module for performing arithmetic operations on data structures.
/// @tparam T the data type (numeric).
/// @tparam D the data structure type (DataStructure).
template <typename T>
class ArithmeticModule {
 public:

  /// @brief Defualt constructor for ArithmeticModule class.
  ArithmeticModule() = default;

  /// @brief Abstract destructor for ArithmeticModule class.
  virtual ~ArithmeticModule() = default;

  /// @brief Add two data structures.
  /// @param a The first data structure.
  /// @param b The second data structure.
  /// @return The result of adding the two data structures.
  virtual DataStructure<T>* add(const DataStructure<T>* a, const DataStructure<T>* b) const = 0;

  /// @brief Subtract one data structure from another.
  /// @param a The data structure to subtract from.
  /// @param b The data structure to subtract.
  /// @return The result of subtracting the second data structure from the first.
  virtual DataStructure<T>* subtract(const DataStructure<T>* a, const DataStructure<T>* b) const = 0;

  /// @brief Multiply two data structures.
  /// @param a The first data structure.
  /// @param b The second data structure.
  /// @return The result of multiplying the two data structures.
  virtual DataStructure<T>* multiply(const DataStructure<T>* a, const DataStructure<T>* b) const = 0;

  /// @brief Multiply a data structure by a scalar.
  /// @param a The data structure.
  /// @param b The scalar value.
  /// @return The result of multiplying the data structure by the scalar.
  virtual DataStructure<T>* multiply(const DataStructure<T>* a, const T b) const = 0;

  /// @brief Divide a data structure by a scalar.
  /// @param a The data structure.
  /// @param b The scalar value.
  /// @return The result of dividing the data structure by the scalar.
  virtual DataStructure<T>* divide(const DataStructure<T>* a, const T b) const = 0;

  /// @brief Check if two data structures are equal.
  /// @param a The first data structure.
  /// @param b The second data structure
  /// @return True if the data structures are equal, false otherwise.
  virtual bool equals(const DataStructure<T>* a, const DataStructure<T>* b) const = 0;
};