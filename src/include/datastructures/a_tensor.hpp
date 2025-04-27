#pragma once

#include <cstdlib>
#include <memory>

#include "datastructures/mml_array.hpp"
#include "datastructures/tensor_concept.hpp"

/*!
 * @brief Abstract class representing a Tensor.
 * @details A tensor is a multi-dimensional ordered set of data.
 * This class is an interface for all implementations of a tensor
 * data structure in ModularML.
 * @tparam T The type of the data contained in the tensor. E.g. int, float,
 * double etc.
 */
template <TensorConcept::Types T>
class Tensor {
 public:
  /// @brief Type alias for the element type
  using value_type = T;

  /// @brief Default constructor for Tensor class.
  Tensor() = default;

  /// @brief Copy constructor for Tensor class.
  /// @param other The tensor to copy.
  Tensor(const Tensor &other) = default;

  /// @brief Move constructor for Tensor class.
  /// @param other The tensor to move.
  Tensor(Tensor &&other) noexcept = default;

  /// @brief Virtual destructor for Tensor class.
  virtual ~Tensor() = default;

  /// @brief Get an element from the tensor using multi-dimensional indices.
  /// @param indices A list of integers representing the indices of the element.
  /// @return A const reference to the element at the given indices.
  virtual const T &operator[](std::initializer_list<size_t> indices) const = 0;

  /// @brief Set an element in the tensor using multi-dimensional indices.
  /// @param indices A list of integers representing the indices of the element.
  /// @return A reference to the element at the given indices.
  virtual T &operator[](std::initializer_list<size_t> indices) = 0;

  /// @brief Get an element from the tensor using multi-dimensional indices.
  /// @param indices An array of integers representing the indices of the
  /// element.
  /// @return A const reference to the element at the given indices.
  virtual const T &operator[](array_mml<size_t> &indices) const = 0;

  /// @brief Check if this tensor is equal to another tensor.
  /// @param other The tensor to compare with.
  /// @return True if the tensors are equal, false otherwise.
  virtual bool operator==(const Tensor<T> &other) const = 0;

  /// @brief Move-Assignment operator.
  /// @param other The tensor to assign.
  /// @return Reference to the assigned tensor.
  virtual Tensor &operator=(Tensor &&other) noexcept = 0;

  /// @brief (Deep) Copy-Assignment operator.
  /// @param other The tensor to assign.
  /// @return Reference to the assigned tensor.
  virtual Tensor &operator=(const Tensor &other) = 0;

  /// @brief Get an element from the tensor using single-dimensional index.
  /// @param index A single integer representing the index of the element.
  /// @return A const reference to the element at the given index.
  virtual const T &operator[](size_t index) const = 0;

  /// @brief Set an element in the tensor using single-dimensional index.
  /// @param index A single integer representing the index of the element.
  /// @return A reference to the element at the given index.
  virtual T &operator[](size_t index) = 0;

  /// @brief Set an element in the tensor using multi-dimensional indices.
  /// @param indices An array of integers representing the indices of the
  /// element.
  /// @return A reference to the element at the given indices.
  virtual T &operator[](array_mml<size_t> &indices) = 0;

  /// @brief Output operator for printing tensor contents.
  /// @param os The output stream.
  /// @param tensor The tensor to output.
  /// @return The modified output stream.
  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    os << tensor.to_string();
    return os;
  }

  /// @brief Get the shape of the tensor.
  /// @return An array of integers representing the shape.
  virtual const array_mml<size_t> &get_shape() const = 0;

  /// @brief Get the total number of elements in the tensor.
  /// @return The total number of elements in the tensor.
  virtual size_t get_size() const = 0;

  /// @brief Fills the tensor with a given value.
  /// @param value The value to fill the tensor with.
  virtual void fill(T value) = 0;

  /// @brief Reverses the buffer of the tensor.
  virtual void reverse_buffer() = 0;

  /// @brief Get a mutable slice of the tensor.
  /// @param slice_indices The indices of the slice.
  /// @return A shared pointer to a slice of the tensor.
  virtual std::shared_ptr<Tensor<T>> slice(
      std::initializer_list<size_t> slice_indices) = 0;

  /// @brief Get a mutable slice of the tensor.
  /// @param slice_indices The indices of the slice.
  /// @return A shared pointer to a slice of the tensor.
  virtual std::shared_ptr<Tensor<T>> slice(
      array_mml<size_t> &slice_indices) = 0;

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor expressed as an array of
  /// integers.
  virtual void reshape(const array_mml<size_t> &new_shape) = 0;

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor expressed as a list of
  /// integers.
  virtual void reshape(std::initializer_list<size_t> new_shape) = 0;

  /// @brief Display the tensor.
  /// @return A string representation of the tensor.
  virtual std::string to_string() const = 0;

  /// @brief Check if the tensor is a matrix.
  /// @return True if the tensor is a matrix (has rank 2), false otherwise.
  virtual bool is_matrix() const = 0;

  /// @brief Transpose the tensor along specified dimensions.
  /// @param dim0 First dimension to transpose (optional).
  /// @param dim1 Second dimension to transpose (optional).
  /// @return A shared pointer to the transposed tensor.
  virtual std::shared_ptr<Tensor<T>> transpose(
      std::optional<size_t> dim0 = std::nullopt,
      std::optional<size_t> dim1 = std::nullopt) const = 0;

  /// @brief Transpose the tensor according to the specified permutation.
  /// @param perm Vector defining the permutation of dimensions.
  /// @return A shared pointer to the transposed tensor.
  virtual std::shared_ptr<Tensor<T>> transpose(
      const std::vector<int> &perm) const = 0;

  /// @brief Reshape and broadcast the tensor to a target shape.
  /// @param target_shape The target shape for broadcasting.
  /// @return A shared pointer to the broadcasted tensor.
  virtual std::shared_ptr<Tensor<T>> broadcast_reshape(
      const array_mml<size_t> &target_shape) const = 0;

  /// @brief Create a copy of the tensor.
  /// @return A shared pointer to the copied tensor.
  virtual std::shared_ptr<Tensor<T>> copy() const = 0;
};