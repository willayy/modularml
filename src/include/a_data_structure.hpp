#pragma once

#include "globals.hpp"

#define ASSERT_ALLOWED_TYPES(T) static_assert(std::is_arithmetic_v<T>, "Data structure type must be an arithmetic type.")

/// @brief  A class representing a n-dimensional data structure used in the Tensor class.
/// @tparam T the type of the data structure.
template <typename T>
class DataStructure {
 public:
  /// @brief Default constructor for DataStructure class.
  DataStructure() = default;

  /// @brief Abstract destructor for DataStructure class.
  virtual ~DataStructure() = default;

  /// @brief Set the data of the data structure.
  /// @param data a vector of data to set in the data structure.
  virtual void set_data(const vec<T> data) = 0;

  /// @brief Set all elements in the data structure to zero.
  virtual void set_zero() = 0;

  /// @brief Get the shape of the data structure.
  /// @return a vector of integers representing the shape.
  virtual const vec<int> &get_shape() const = 0;

  /// @brief Get the shape as a string. E.g. [2, 3, 4].
  /// @return a string representation of the shape.
  virtual string get_shape_str() const = 0;

  /// @brief Gets the amount of elements in the data structure.
  /// @return the size of the data structure as an integer.
  virtual int get_size() const = 0;

  /// @brief Compares the data structure with another data structure for equality.
  /// @param other the other data structure to compare with.
  /// @return true if the data structures are equal, false otherwise.
  virtual bool equals(const DataStructure<T> &other) const = 0;

  /// @brief Get an element from the data structure.
  /// @param indices a vector of integers representing the indices of the data structure.
  /// @return a value from the data structure.
  virtual T get(const vec<int> &indices) const = 0;

  /// @brief Set an element in the data structure.
  /// @param indices a vector of integers representing the indices of the data structure.
  /// @param value the value to set in the data structure.
  virtual void set(const vec<int> &indices, T value) = 0;
};
