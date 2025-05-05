#pragma once

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
#include <vector>  // IWYU pragma: keep

/// @brief Array class mimicking the std::array class but without the size being
/// a template parameter.
/// @tparam T the type of the array.
template <typename T>
class array_mml {
 public:

  static_assert(std::is_arithmetic_v<T>,
                "array_mml must have an arithmetic type.");

  /// @brief Default constructor for array_mml class.
  array_mml() : data(std::make_unique<T[]>(0)), d_size(0) {}

  /// @brief Constructor for array_mml class.
  /// @param size The size of the array.
  explicit array_mml(size_t size);

  /// @brief Constructor for array_mml class.
  /// @param data The data to set in the array as a initializer list.
  array_mml(std::initializer_list<T> data);

  /// @brief Constructor for array_mml class.
  /// @param data The pointer to the data to set in the array.
  /// @param size The size of the data.
  array_mml(std::shared_ptr<T[]> data, size_t size);

  /// @brief Constructor for array_mml class.
  /// @param data The data to set in the array.
  explicit array_mml(std::vector<T> &data);

  /// @brief Copy constructor using a std::vector.
  /// @param data The data to std::copy.
  explicit array_mml(const std::vector<T> &data);

  /// @brief Copy constructor using another array.
  array_mml(const array_mml &other);

  /// @brief Move constructor for array_mml class.
  array_mml(array_mml &&other) noexcept;

  /// @brief Get the size of the array, the number of elements in the array.
  /// @return The size of the array.
  size_t size() const;

  /// @brief Get an element from the array using a single-dimensional index.
  /// @param index The index of the element to get.
  /// @return The element at the given index.
  T &operator[](size_t index);

  /// @brief Get an element from the array using a single-dimensional index.
  /// @param index The index of the element to get.
  /// @return The element at the given index.
  const T &operator[](size_t index) const;

  /// @brief Move assignment operator.
  /// @param other The array to std::move.
  /// @return The moved array.
  array_mml &operator=(array_mml &&other) noexcept = default;

  /// @brief Copy assignment operator.
  /// @param other The array to std::copy.
  /// @return The copied array.
  array_mml &operator=(const array_mml &other);

  /// @brief Get a subarray from the array.
  /// @param start The start index of the subarray.
  /// @param end The end index of the subarray.
  /// @return The subarray.
  array_mml subarray(size_t start, size_t end) const;

  /// @brief Equality operator.
  /// @param other The array to compare with.
  /// @return True if the arrays are std::equal, false otherwise.
  bool operator==(const array_mml &other) const;

  /// @brief Convert the array to a std::string.
  /// @return The std::string representation of the array.
  std::string to_string() const;

  /// @brief Output stream operator.
  /// @param os The output stream.
  /// @param arr The array to output.
  /// @return The output stream.
  friend std::ostream &operator<<(std::ostream &os, const array_mml<T> &arr) {
    os << arr.to_string();
    return os;
  }

  /// @brief Get an iterator to the beginning of the array.
  /// @return An iterator to the beginning of the array.
  T *begin();

  /// @brief Get a const iterator to the beginning of the array.
  /// @return A const iterator to the beginning of the array.
  const T *begin() const;

  /// @brief Get an iterator to the end of the array.
  /// @return An iterator to the end of the array.
  T *end();

  /// @brief Get a const iterator to the end of the array.
  /// @return A const iterator to the end of the array.
  const T *end() const;

  /// @brief Get a pointer to the underlying data.
  /// @return A pointer to the underlying data.
  T *get();

  /// @brief Get a const pointer to the underlying data.
  /// @return A const pointer to the underlying data.
  const T *get() const;

  /// @brief Fill the array with a given value.
  /// @param value The value to fill the array with.
  void fill(const T &value);

 private:
  std::shared_ptr<T[]> data;
  size_t d_size;
};

#include "../datastructures/mml_array.tpp"