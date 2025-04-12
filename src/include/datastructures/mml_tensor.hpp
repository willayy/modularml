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

#include "datastructures/a_tensor.hpp"

/*!
 * @brief A Tensor<T> implementation using an underlying
 * fixed size 1D array with row-major offsets for
 * multi-dimensional indexing.
 * @tparam T The type of the data contained in the tensor.
 * Allows for arithmetic types.
 */
template <typename T>
class Tensor_mml : public Tensor<T> {
 public:
  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(const std::initializer_list<size_t> shape,
                      const size_t jump_indexes = 0,
                      const size_t jump_columns = 0, const size_t jump_rows = 0,
                      const bool sliced = false);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(const std::initializer_list<size_t> shape,
                      const std::initializer_list<T> data,
                      const size_t jump_indexes = 0,
                      const size_t jump_columns = 0, const size_t jump_rows = 0,
                      const bool sliced = false);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(const array_mml<size_t> &shape,
                      const size_t jump_indexes = 0,
                      const size_t jump_columns = 0, const size_t jump_rows = 0,
                      const bool sliced = false);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(const array_mml<size_t> &shape, const array_mml<T> &data,
                      const size_t jump_indexes = 0,
                      const size_t jump_columns = 0, const size_t jump_rows = 0,
                      const bool sliced = false);

  /// @brief Destructor for Tensor_mml class.
  ~Tensor_mml() = default;

  /// @brief Move constructor for Tensor_mml class.
  Tensor_mml(Tensor_mml &&other) noexcept;

  /// @brief Copy constructor for Tensor_mml class.
  Tensor_mml(const Tensor_mml &other);

  /// @brief Get the raw 1D data of the tensor.
  /// @return The data of the tensor.
  const array_mml<T> &get_data() const;

  /// Ovveridden methods from the base class
  Tensor<T> &operator=(const Tensor<T> &other) override;
  Tensor<T> &operator=(Tensor<T> &&other) noexcept override;
  std::string to_string() const override;
  std::shared_ptr<Tensor<T>> copy() const override;
  void reverse_buffer() override;
  std::shared_ptr<Tensor<T>> slice(
      std::initializer_list<size_t> slice_indices) override;
  std::shared_ptr<Tensor<T>> slice(array_mml<size_t> &slice_indices) override;
  void reshape(const array_mml<size_t> &new_shape) override;
  void reshape(std::initializer_list<size_t> new_shape) override;
  bool is_matrix() const override;
  bool matrix_match(const Tensor<T> &other) const override;
  bool operator==(const Tensor<T> &other) const override;
  bool operator!=(const Tensor<T> &other) const override;
  const array_mml<size_t> &get_shape() const override;
  const array_mml<size_t> &get_offsets() const;
  size_t get_size() const override;
  const T &operator[](array_mml<size_t> &indices) const override;
  T &operator[](array_mml<size_t> &indices) override;
  const T &operator[](std::initializer_list<size_t> indices) const override;
  T &operator[](std::initializer_list<size_t> indices) override;
  const T &operator[](size_t index) const override;
  T &operator[](size_t index) override;
  void fill(T value) override;
  std::shared_ptr<Tensor<T>> transpose(
      std::optional<size_t> dim0 = std::nullopt,
      std::optional<size_t> dim1 = std::nullopt) const override;
  std::shared_ptr<Tensor<T>> broadcast_reshape(
      const array_mml<size_t> &target_shape) const override;

 private:
  array_mml<T> data;
  array_mml<size_t> shape;
  array_mml<size_t> indices_offsets;
  bool sliced;
  size_t jump_indexes;
  size_t jump_rows;
  size_t jump_columns;
  size_t size;

  // Helper methods
  size_t compute_size() const;
  array_mml<size_t> compute_indices_offsets() const;
  bool valid_shape(const array_mml<size_t> &new_shape) const;
  bool valid_indices(const array_mml<size_t> &indices) const;
  bool valid_index(size_t index) const;
  bool valid_slice_indices(const array_mml<size_t> &slice_indices) const;
  bool valid_broadcast_reshape_size(
      const array_mml<size_t> &target_shape) const;
  size_t indices_to_1d_index(array_mml<size_t> indices) const;
  size_t index_to_offset_1d_index(size_t index) const;
};

template <typename T>
[[deprecated("Use TensorFactory instead")]]
std::shared_ptr<Tensor<T>> tensor_mml_p(
    const std::initializer_list<size_t> shape);

template <typename T>
[[deprecated("Use TensorFactory instead")]]
std::shared_ptr<Tensor<T>> tensor_mml_p(
    const std::initializer_list<size_t> shape,
    const std::initializer_list<T> data);

#include "../datastructures/mml_tensor.tpp"
