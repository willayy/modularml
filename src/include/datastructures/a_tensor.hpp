#pragma once

#include <cstdlib>
#include <memory>

#include "datastructures/mml_array.hpp"

#define ASSERT_ALLOWED_TYPE_T(T)         \
  static_assert(std::is_arithmetic_v<T>, \
                "Tensor must have an arithmetic type.");

/*!
 * @brief Abstract class representing a Tensor.
 * @details A tensor is a multi-dimensional ordered set of data.
 * This class is an interface for all implemnations of a tensor
 * data structure in ModularML.
 * @tparam T the type of the data contained in the tensor. E.g. int, float,
 * double etc.
 */
template <typename T>
class Tensor {
 public:
  using value_type = T;

  /// @brief Default constructor for Tensor class.
  Tensor() = default;

  /// @brief Copy constructor for Tensor class.
  /// @param other The tensor to std::copy.
  Tensor(const Tensor &other) = default;

  /// @brief Move constructor for Tensor class.
  /// @param other The tensor to std::move.
  Tensor(Tensor &&other) noexcept = default;

  /// @brief Destructor for Tensor class.
  virtual ~Tensor() = default;

  /// @brief Get an element from the tensor using multi-dimensional indices.
  /// @param indices A std::vector of integers representing the indices of the
  /// element.
  /// @return An element at the given indices.
  virtual const T &operator[](std::initializer_list<size_t> indices) const = 0;

  /// @brief Set an element in the tensor using multi-dimensional indices.
  /// @param indices A std::vector of integers representing the indices of the
  /// element.
  /// @return An element at the given indices.
  virtual T &operator[](std::initializer_list<size_t> indices) = 0;

  ///@brief Get an element from the tensor using multi-dimensional indices.
  ///@param indices A std::vector of integers representing the indices of the
  /// element.
  ///@return An element at the given indices.
  virtual const T &operator[](array_mml<size_t> &indices) const = 0;

  ///@brief Check if this tensor is std::equal to another tensor.
  ///@param other The tensor to compare with.
  ///@return True if the tensors are std::equal, false otherwise.*/
  virtual bool operator==(const Tensor<T> &other) const = 0;

  ///@brief Move-Assignment operator.
  ///@param other The tensor to assign.
  ///@return The moved tensor.
  virtual Tensor &operator=(Tensor &&other) noexcept = 0;

  /// @brief (Deep) Copy-Assigment operator.
  /// @param other The tensor to assign.
  /// @return The copied tensor.
  virtual Tensor &operator=(const Tensor &other) = 0;

  ///@brief Get an element from the tensor using singel-dimensional index.
  ///@param index A single integer representing the index of the element.
  ///@return The element at the given indices.*/
  virtual const T &operator[](size_t index) const = 0;

  ///@brief Set an element in the tensor using single-dimensional index.
  ///@param index A single integer representing the index of the element.
  ///@return The tensor with the element get_mutable_elem.*/
  virtual T &operator[](size_t index) = 0;

  ///@brief Set an element in the tensor using multi-dimensional indices.
  ///@param indices A std::vector of integers representing the indices of the
  /// element.
  ///@return An element at the given indices.
  virtual T &operator[](array_mml<size_t> &indices) = 0;

  /// @brief Get the shape as a std::string.
  /// @return A std::string representation of the shape. E.g. [2, 3, 4].
  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    os << tensor.to_string();
    return os;
  }

  virtual array_mml<T> &get_raw_data() = 0;

  /// @brief Get the shape of the tensor.
  /// @return A std::vector of integers representing the shape.
  virtual const array_mml<size_t> &get_shape() const = 0;

  /// @brief Get the the total number of elements in the tensor.
  /// @return The total number of elements in the tensor.
  virtual size_t get_size() const = 0;

  /// @brief Fills the tensor with a given value.
  /// @param value The value to fill the tensor with.
  virtual void fill(T value) = 0;

  /// @brief Reverses the buffer of the tensor.
  virtual void reverse_buffer() = 0;

  /// @brief Get a mutable slice of the tensor.
  /// @param slice_indices The indices of the slice.
  /// @return A slice of the tensor.
  virtual std::shared_ptr<Tensor<T>> slice(
      std::initializer_list<size_t> slice_indices) = 0;

  /// @brief Get a mutable slice of the tensor.
  /// @param slice_indices The indices of the slice.
  /// @return A slice of the tensor.
  virtual std::shared_ptr<Tensor<T>> slice(
      array_mml<size_t> &slice_indices) = 0;

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor expressed as a list of
  /// integers.
  virtual void reshape(const array_mml<size_t> &new_shape) = 0;

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor expressed as a list of
  /// integers.
  virtual void reshape(std::initializer_list<size_t> new_shape) = 0;

  /// @brief Display the tensor.
  /// @return A std::string representation of the tensor.
  virtual std::string to_string() const = 0;

  /// @brief Check if the tensor is a matrix.
  /// @return True if the tensor is a matrix (has rank 2), false otherwise.
  virtual bool is_matrix() const = 0;

  virtual std::shared_ptr<Tensor<T>> transpose(
      std::optional<size_t> dim0 = std::nullopt,
      std::optional<size_t> dim1 = std::nullopt) const = 0;

  virtual std::shared_ptr<Tensor<T>> transpose(
      const std::vector<int> &perm) const = 0;

  virtual std::shared_ptr<Tensor<T>> broadcast_reshape(
      const array_mml<size_t> &target_shape) const = 0;

  /// @brief Method way to get a std::copy of the tensor.
  /// @return A shared pointer to the copied tensor.
  virtual std::shared_ptr<Tensor<T>> copy() const = 0;
};