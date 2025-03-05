#pragma once

#include "globals.hpp"

// Assert size_d is a positive integer.


/// @brief Array class mimicking the std::array class but without the size being a template parameter.
/// @tparam T the type of the array.
template <typename T>
class array_mml {
 public:
  /// @brief Constructor for array_ml class.
  /// @param size The size of the array.
  explicit array_mml(uint64_t size) : data(make_unique<T[]>(size)), d_size(size) {}

  explicit array_mml(initializer_list<T> data) : data(make_unique<T[]>(data.size())), d_size(data.size()) {
    copy(data.begin(), data.end(), this->data.get());
  }

  explicit array_mml(vector<T>& data) : data(make_unique<T[]>(data.size())), d_size(data.size()) {
    copy(data.begin(), data.end(), this->data.get());
  }

  explicit array_mml(const vector<T>& data) : data(make_unique<T[]>(data.size())), d_size(data.size()) {
    copy(data.begin(), data.end(), this->data.get());
  }

  /// @brief Copy constructor for array_ml class.
  array_mml(const array_mml& other): data(make_unique<T[]>(other.d_size)), d_size(other.d_size) {
    copy(other.data.get(), other.data.get() + other.d_size, this->data.get());
  }

  /// @brief Move constructor for array_ml class.
  array_mml(array_mml&& other) noexcept = default;

  /// @brief Destructor for array_ml class.
  ~array_mml() = default;

  /// @brief Get the size of the array, the number of elements in the array.
  /// @return The size of the array.
  uint64_t size() const {
    return this->d_size;
  }

  /// @brief Get an element from the array using a single-dimensional index.
  /// @param index The index of the element to get.
  /// @return The element at the given index.
  T& operator[](int index) {
    if (index < 0 || index >= this->d_size) {
      throw out_of_range("Invalid array_mml index");
    } else {
      return this->data[index];
    }
  }

  /// @brief Get an element from the array using a single-dimensional index.
  /// @param index The index of the element to get.
  /// @return The element at the given index.
  const T& operator[](int index) const {
    if (index < 0 || index >= this->d_size) {
      throw out_of_range("Invalid array_mml index");
    } else {
      return this->data[index];
    }
  }

 private:
  unique_ptr<T[]> data; // NOSONAR - unique_ptr is the correct data structure to use here, we cant use std::array because it requires the size to be a template parameter.
  uint64_t d_size;

};