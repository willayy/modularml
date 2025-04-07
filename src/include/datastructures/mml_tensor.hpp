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

/*!
 * @brief A Tensor<T> implementation using an underlying
 * fixed size 1D array with row-major offsets for
 * multi-dimensional indexing.
 * @tparam T The type of the data contained in the tensor.
 * Allows for arithmetic types.
 */
template <typename T> class Tensor_mml : public Tensor<T> {
public:
  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(
      const std::initializer_list<uli> shape,
      std::optional<array_mml<uli>> slice_offsets = std::nullopt);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(
      const std::initializer_list<uli> shape,
      const std::initializer_list<T> data,
      std::optional<array_mml<uli>> slice_offsets = std::nullopt);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(
      const array_mml<uli> &shape,
      std::optional<array_mml<uli>> slice_offsets = std::nullopt);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  [[deprecated("Use TensorFactory instead")]]
  explicit Tensor_mml(
      const array_mml<uli> &shape, const array_mml<T> &data,
      std::optional<array_mml<uli>> slice_offsets = std::nullopt);

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
  std::shared_ptr<Tensor<T>>
  slice(std::initializer_list<uli> slice_indices) override;
  std::shared_ptr<Tensor<T>> slice(array_mml<uli> &slice_indices) override;
  void reshape(const array_mml<uli> &new_shape) override;
  void reshape(std::initializer_list<uli> new_shape) override;
  bool is_matrix() const override;
  bool matrix_match(const Tensor<T> &other) const override;
  bool operator==(const Tensor<T> &other) const override;
  bool operator!=(const Tensor<T> &other) const override;
  const array_mml<uli> &get_shape() const override;
  const array_mml<uli> &get_offsets() const;
  uli get_size() const override;
  const T &operator[](array_mml<uli> &indices) const override;
  T &operator[](array_mml<uli> &indices) override;
  const T &operator[](std::initializer_list<uli> indices) const override;
  T &operator[](std::initializer_list<uli> indices) override;
  const T &operator[](uli index) const override;
  T &operator[](uli index) override;
  void fill(T value) override;
  std::shared_ptr<Tensor<T>>
  transpose(std::optional<uli> dim0 = std::nullopt,
            std::optional<uli> dim1 = std::nullopt) const override;
  std::shared_ptr<Tensor<T>>
  broadcast_to(const array_mml<uli> &target_shape) const override;

private:
  array_mml<T> data;
  array_mml<uli> shape;
  array_mml<uli> indices_offsets;
  std::optional<array_mml<uli>> slice_offsets;
  uli size;

  // Helper methods
  uli compute_size() const;
  array_mml<uli> compute_indices_offsets() const;
  array_mml<uli> compute_slice_offsets(array_mml<uli> &slice_indices_size,
                                       array_mml<uli> &slice_shape) const;
  bool valid_shape(const array_mml<uli> &new_shape) const;
  bool valid_indices(const array_mml<uli> &indices) const;
  bool valid_index(uli index) const;
  bool valid_slice_indices(const array_mml<uli> &slice_indices) const;
  uli indices_to_1d_index(array_mml<uli> indices) const;
  uli index_to_slice_index(uli index) const;
  bool is_broadcastable_to(const array_mml<uli> &target_shape) const;
};

template <typename T>
[[deprecated("Use TensorFactory instead")]]
std::shared_ptr<Tensor<T>> tensor_mml_p(const std::initializer_list<uli> shape);

template <typename T>
[[deprecated("Use TensorFactory instead")]]
std::shared_ptr<Tensor<T>> tensor_mml_p(const std::initializer_list<uli> shape,
                                        const std::initializer_list<T> data);

#include "../datastructures/mml_tensor.tpp"
