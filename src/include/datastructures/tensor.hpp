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
#include <vector>

#include "datastructures/mml_array.hpp"

/*!
 * @brief A Tensor<T> implementation using an underlying
 * fixed size 1D array with row-major offsets for
 * multi-dimensional indexing.
 * @tparam T The type of the data contained in the tensor.
 * Allows for arithmetic types.
 */
template <typename T> class Tensor {
public:
  using value_type = T;

  /// @brief Constructor for Tensor class.
  /// @param shape The shape of the tensor.
  explicit Tensor(
      const std::initializer_list<size_t> shape,
      std::optional<array_mml<size_t>> slice_offsets = std::nullopt);

  /// @brief Constructor for Tensor class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor(
      const std::initializer_list<size_t> shape,
      const std::initializer_list<T> data,
      std::optional<array_mml<size_t>> slice_offsets = std::nullopt);

  /// @brief Constructor for Tensor class.
  /// @param shape The shape of the tensor.
  explicit Tensor(
      const array_mml<size_t> &shape,
      std::optional<array_mml<size_t>> slice_offsets = std::nullopt);

  /// @brief Constructor for Tensor class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor(
      const array_mml<size_t> &shape, const array_mml<T> &data,
      std::optional<array_mml<size_t>> slice_offsets = std::nullopt);

  /// @brief Destructor for Tensor class.
  ~Tensor() = default;

  /// @brief Move constructor for Tensor class.
  Tensor(Tensor &&other) noexcept;

  /// @brief Copy constructor for Tensor class.
  Tensor(const Tensor &other);

  /// @brief Get the raw 1D data of the tensor.
  /// @return The data of the tensor.
  const array_mml<T> &get_data() const;

  /// Ovveridden methods from the base class
  Tensor<T> &operator=(const Tensor<T> &other);
  Tensor<T> &operator=(Tensor<T> &&other) noexcept;
  std::string to_string() const;
  std::shared_ptr<Tensor<T>> copy() const;
  void reverse_buffer();
  std::shared_ptr<Tensor<T>>
  slice(std::initializer_list<size_t> slice_indices);
  std::shared_ptr<Tensor<T>> slice(array_mml<size_t> &slice_indices);
  void reshape(const array_mml<size_t> &new_shape);
  void reshape(std::initializer_list<size_t> new_shape);
  bool is_matrix() const;
  bool matrix_match(const Tensor<T> &other) const;
  bool operator==(const Tensor<T> &other) const;
  bool operator!=(const Tensor<T> &other) const;
  const array_mml<size_t> &get_shape() const;
  const array_mml<size_t> &get_offsets() const;
  size_t get_size() const;
  const T &operator[](array_mml<size_t> &indices) const;
  T &operator[](array_mml<size_t> &indices);
  const T &operator[](std::initializer_list<size_t> indices) const;
  T &operator[](std::initializer_list<size_t> indices);
  const T &operator[](size_t index) const;
  T &operator[](size_t index);
  void fill(T value);
  std::shared_ptr<Tensor<T>>
  transpose(std::optional<size_t> dim0 = std::nullopt,
            std::optional<size_t> dim1 = std::nullopt) const;
  std::shared_ptr<Tensor<T>>
  broadcast_to(const array_mml<size_t> &target_shape) const;

private:
  array_mml<T> data;
  array_mml<size_t> shape;
  array_mml<size_t> indices_offsets;
  std::optional<array_mml<size_t>> slice_offsets;
  size_t size;

  // Helper methods
  size_t compute_size() const;
  array_mml<size_t> compute_indices_offsets() const;
  array_mml<size_t> compute_slice_offsets(array_mml<size_t> &slice_indices_size,
                                          array_mml<size_t> &slice_shape) const;
  bool valid_shape(const array_mml<size_t> &new_shape) const;
  bool valid_indices(const array_mml<size_t> &indices) const;
  bool valid_index(size_t index) const;
  bool valid_slice_indices(const array_mml<size_t> &slice_indices) const;
  size_t indices_to_1d_index(array_mml<size_t> indices) const;
  size_t index_to_slice_index(size_t index) const;
  bool is_broadcastable_to(const array_mml<size_t> &target_shape) const;

};

#define _TENSOR(DT) template class Tensor<DT>;
