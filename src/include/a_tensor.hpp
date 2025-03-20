#pragma once

#include "array_mml.hpp"
#include "globals.hpp"

#define ASSERT_ALLOWED_TYPE_T(T) static_assert(std::is_arithmetic_v<T>, "Tensor must have an arithmetic type.");

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

  /// @brief Default constructor for Tensor class.
  Tensor() = default;

  /// @brief Copy constructor for Tensor class.
  /// @param other The tensor to copy.
  Tensor(const Tensor &other) = default;

  /// @brief Move constructor for Tensor class.
  /// @param other The tensor to move.
  Tensor(Tensor &&other) noexcept = default;

  /// @brief Destructor for Tensor class.
  virtual ~Tensor() = default;

  /// @brief Check if this tensor is not equal to another tensor.
  /// @param other The tensor to compare with.
  /// @return True if the tensors are not equal, false otherwise.
  virtual bool operator!=(const Tensor<T> &other) const = 0;  // NOSONAR - Fair point but, We choose this to enable different tensor implementation but using the same interface.

  /// @brief Get an element from the tensor using multi-dimensional indices.
  /// @param indices A vector of integers representing the indices of the element.
  /// @return An element at the given indices.
  virtual const T &operator[](initializer_list<uli> indices) const = 0;

  /// @brief Set an element in the tensor using multi-dimensional indices.
  /// @param indices A vector of integers representing the indices of the element.
  /// @return An element at the given indices.
  virtual T &operator[](initializer_list<uli> indices) = 0;

  ///@brief Get an element from the tensor using multi-dimensional indices.
  ///@param indices A vector of integers representing the indices of the element.
  ///@return An element at the given indices.
  virtual const T &operator[](array_mml<uli> &indices) const = 0;

  ///@brief Check if this tensor is equal to another tensor.
  ///@param other The tensor to compare with.
  ///@return True if the tensors are equal, false otherwise.*/
  virtual bool operator==(const Tensor<T> &other) const = 0;  // NOSONAR - Fair point but, We choose this to enable different tensor implementation but using the same interface.

  ///@brief Move-Assignment operator.
  ///@param other The tensor to assign.
  ///@return The moved tensor.
  virtual Tensor &operator=(Tensor &&other) noexcept = 0;  // NOSONAR - Fair point but, We choose this to enable different tensor implementation but using the same interface.

  /// @brief (Deep) Copy-Assigment operator.
  /// @param other The tensor to assign.
  /// @return The copied tensor.
  virtual Tensor &operator=(const Tensor &other) = 0;  // NOSONAR - Fair point but, We choose this to enable different tensor implementation but using the same interface.

  ///@brief Get an element from the tensor using singel-dimensional index.
  ///@param index A single integer representing the index of the element.
  ///@return The element at the given indices.*/
  virtual const T &operator[](uli index) const = 0;

  ///@brief Set an element in the tensor using single-dimensional index.
  ///@param index A single integer representing the index of the element.
  ///@return The tensor with the element get_mutable_elem.*/
  virtual T &operator[](uli index) = 0;

  ///@brief Set an element in the tensor using multi-dimensional indices.
  ///@param indices A vector of integers representing the indices of the element.
  ///@return An element at the given indices.
  virtual T &operator[](array_mml<uli> &indices) = 0;

  /// @brief Get the shape as a string.
  /// @return A string representation of the shape. E.g. [2, 3, 4].
  friend ostream &operator<<(ostream &os, const Tensor<T> &tensor) {
    os << tensor.to_string();
    return os;
  }

  /// @brief Get the shape of the tensor.
  /// @return A vector of integers representing the shape.
  virtual const array_mml<uli> &get_shape() const = 0;

  /// @brief Get the the total number of elements in the tensor.
  /// @return The total number of elements in the tensor.
  virtual uli get_size() const = 0;

  /// @brief Fills the tensor with a given value.
  /// @param value The value to fill the tensor with.
  virtual void fill(T value) = 0;

  /// @brief Flips the tensor along a given dimension.
  /// @param dim The dimension to flip along.
  virtual void flip(uli dim) = 0;

  /// @brief Get a mutable slice of the tensor.
  /// @param slice_indices The indices of the slice.
  /// @return A slice of the tensor.
  virtual shared_ptr<Tensor<T>> slice(initializer_list<uli> slice_indices) = 0;

  /// @brief Get a mutable slice of the tensor.
  /// @param slice_indices The indices of the slice.
  /// @return A slice of the tensor.
  virtual shared_ptr<Tensor<T>> slice(array_mml<uli> &slice_indices) = 0;

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor expressed as a list of integers.
  virtual void reshape(const array_mml<uli> &new_shape) = 0;

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor expressed as a list of integers.
  virtual void reshape(initializer_list<uli> new_shape) = 0;

  /// @brief Display the tensor.
  /// @return A string representation of the tensor.
  virtual string to_string() const = 0;

  /// @brief Check if the tensor is a matrix.
  /// @return True if the tensor is a matrix (has rank 2), false otherwise.
  virtual bool is_matrix() const = 0;

  /// @brief Check if the tensor-matrix matches another matrix. Assumes the tensor is a matrix.
  /// @param other The other matrix to compare with.
  /// @return True if the tensor-matrix matches the other matrix, false otherwise.
  virtual bool matrix_match(const Tensor<T> &other) const = 0;

  /// @brief Method way to get a copy of the tensor.
  /// @return A shared pointer to the copied tensor.
  virtual shared_ptr<Tensor<T>> copy() const = 0;
};