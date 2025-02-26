#pragma once

#include <cmath>

#include "globals.hpp"

#define ASSERT_ALLOWED_TYPES(T) static_assert(std::is_arithmetic_v<T>, "Data structure type must be an arithmetic type.")

/// @brief A class creating an interface for one dimensional data structures.
/// @tparam T the type of the data structure.
template <typename T>
class DataStructure {
 public:
  /// @brief Default constructor for DataStructure class.
  DataStructure() = default;

  /// @brief Copy constructor for DataStructure class.
  DataStructure(const DataStructure& other) = default;

  /// @brief Move constructor for DataStructure class.
  DataStructure(DataStructure&& other) noexcept = default;

  /// @brief Abstract destructor for DataStructure class.
  virtual ~DataStructure() = default;

  /// @brief Set the data of the data structure using a vector of data.
  /// @param data a vector of data.
  virtual void set_data(const vector<T> data) = 0;

  /// @brief Set all elements in the data structure to zero.
  virtual void set_zero() = 0;

  /// @brief Gets the amount of elements in the data structure.
  /// @return the size of the data structure as an integer.
  virtual int get_size() const = 0;

  /// @brief Get the raw data of the data structure as a vector.
  /// @return a vector data.
  const virtual vector<T>& get_raw_data() const = 0;

  /// @brief Get an element from the data structure.
  /// @param index an integer representing the index of the data structure.
  /// @return a value from the data structure.
  virtual const T& get_elem(int index) const = 0;

  /// @brief Set an element in the data structure.
  /// @param index an integer representing the index of the data structure.
  virtual T& get_mutable_elem(int index) = 0;

  /// @brief Clone the data structure.
  /// @return a unique pointer to a new data structure with the same data.
  virtual unique_ptr<DataStructure<T>> clone() const = 0;

  /// @brief Array subscript operator for getting an element from the data structure.
  /// @param index an integer representing the index of the data structure.
  /// @return a value from the data structure.
  const T& operator[](int index) const {
    return get_elem(index);
  }

  /// @brief Array subscript operator for setting an element in the data structure.
  /// @param index an integer representing the index of the data structure.
  /// @return a value from the data structure.
  T& operator[](int index) {
    return get_mutable_elem(index);
  }

};
