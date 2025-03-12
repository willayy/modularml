#pragma once

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

#include "globals.hpp"

/// @brief Array class mimicking the std::array class but without the size being a template parameter.
/// @tparam T the type of the array.
template <typename T>
class array_mml {
 public:

  /// @brief Default constructor for array_mml class.
  array_mml() : data(make_unique<T[]>(0)), d_size(0) {}

  /// @brief Constructor for array_mml class.
  /// @param size The size of the array.
  explicit array_mml(uint64_t size) : data(make_unique<T[]>(size)), d_size(size) {}

  /// @brief Constructor for array_mml class.
  /// @param data The data to set in the array.
  explicit array_mml(initializer_list<T> data) : data(make_unique<T[]>(data.size())), d_size(data.size()) {
    copy(data.begin(), data.end(), this->data.get());
  }

  /// @brief Constructor for array_mml class.
  /// @param data The data to set in the array.
  explicit array_mml(vector<T>& data) : data(make_unique<T[]>(data.size())), d_size(data.size()) {
    copy(data.begin(), data.end(), this->data.get());
  }

  /// @brief Copy constructor for array_mml class.
  /// @param data The data to copy.
  explicit array_mml(const vector<T>& data) : data(make_unique<T[]>(data.size())), d_size(data.size()) {
    copy(data.begin(), data.end(), this->data.get());
  }

  /// @brief Copy constructor for array_mml class.
  array_mml(const array_mml& other) : data(make_unique<T[]>(other.d_size)), d_size(other.d_size) {
    copy(other.data.get(), other.data.get() + other.d_size, this->data.get());
  }

  /// @brief Move constructor for array_mml class.
  array_mml(array_mml&& other) noexcept : data(move(other.data)), d_size(other.d_size) {
    other.d_size = 0;
  }

  /// @brief Destructor for array_mml class.
  ~array_mml() = default;

  /// @brief Get the size of the array, the number of elements in the array.
  /// @return The size of the array.
  uint64_t size() const {
    return this->d_size;
  }

  /// @brief Get an element from the array using a single-dimensional index.
  /// @param index The index of the element to get.
  /// @return The element at the given index.
  T& operator[](uint64_t index) {
    if (index >= this->d_size) {
      throw out_of_range("Invalid array_mml index");
    } else {
      return this->data[index];
    }
  }

  /// @brief Get an element from the array using a single-dimensional index.
  /// @param index The index of the element to get.
  /// @return The element at the given index.
  const T& operator[](uint64_t index) const {
    if (index >= this->d_size) {
      throw out_of_range("Invalid array_mml index");
    } else {
      return this->data[index];
    }
  }

  /// @brief Move assignment operator.
  /// @param other The array to move.
  /// @return The moved array.
  array_mml& operator=(array_mml&& other) noexcept = default;

  /// @brief Copy assignment operator.
  /// @param other The array to copy.
  /// @return The copied array.
  array_mml& operator=(const array_mml& other) {
    if (this != &other) {
      copy(other.begin(), other.end(), this->data.get());
      this->d_size = other.d_size;
    }
    return *this;
  }

  array_mml subarray(uint64_t start, uint64_t end) const {
    if (start >= this->d_size || end > this->d_size || start > end) {
      throw out_of_range("Invalid array_mml index");
    }
    array_mml new_array(end - start);
    copy(this->data.get() + start, this->data.get() + end, new_array.data.get());
    return new_array;
  }

  /// @brief Equality operator.
  /// @param other The array to compare with.
  /// @return True if the arrays are equal, false otherwise.
  bool operator==(const array_mml& other) const {
    if (this->d_size != other.d_size) {
      return false;
    }
    return equal(this->begin(), this->end(), other.begin());
  }

  /// @brief Inequality operator.
  /// @param other The array to compare with.
  /// @return True if the arrays are not equal, false otherwise.
  bool operator!=(const array_mml& other) const {
    return !(*this == other);
  }

  string to_string() const {
    string str = "[";
    // if longer than 50 print first 10 then ... then last 10
    if (this->size() > 50) {
      for (uint64_t i = 0; i < 10; i++) {
        str += std::to_string(this->data[i]);
        str += ", ";
      }
      str += "..., ";
      for (uint64_t i = this->size() - 10; i < this->size(); i++) {
        str += std::to_string(this->data[i]);
        if (i != this->size() - 1) {
          str += ", ";
        }
      }
    } else {
      for (uint64_t i = 0; i < this->size(); i++) {
        str += std::to_string(this->data[i]);
        if (i != this->size() - 1) {
          str += ", ";
        }
      }
    }
    str += "]";
    return str;
  }

  friend ostream& operator<<(ostream& os, const array_mml<T>& arr) {
    os << arr.to_string();
    return os;
  }

  /// @brief Get an iterator to the beginning of the array.
  /// @return An iterator to the beginning of the array.
  T* begin() {
    return this->data.get();
  }

  /// @brief Get a const iterator to the beginning of the array.
  /// @return A const iterator to the beginning of the array.
  const T* begin() const {
    return this->data.get();
  }

  /// @brief Get an iterator to the end of the array.
  /// @return An iterator to the end of the array.
  T* end() {
    return this->data.get() + this->d_size;
  }

  /// @brief Get a const iterator to the end of the array.
  /// @return A const iterator to the end of the array.
  const T* end() const {
    return this->data.get() + this->d_size;
  }

  /// @brief Get a pointer to the underlying data.
  /// @return A pointer to the underlying data.
  T* get() {
    return this->data.get();
  }

  /// @brief Get a const pointer to the underlying data.
  /// @return A const pointer to the underlying data.
  const T* get() const {
    return this->data.get();
  }

  /// @brief Fill the array with a given value.
  /// @param value The value to fill the array with.
  void fill(const T& value) {
    std::fill(this->begin(), this->end(), value);
  }

 private:
  unique_ptr<T[]> data;  // NOSONAR - unique_ptr is the correct data structure to use here, we cant use std::array because it requires the size to be a template parameter.
  uint64_t d_size;
};