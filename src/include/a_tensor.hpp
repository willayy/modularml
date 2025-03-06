#pragma once

#include "array_mml.hpp"
#include "globals.hpp"

#define ASSERT_ALLOWED_TYPE_T(T) static_assert(std::is_arithmetic_v<T>, "Tensor must have an arithmetic type.");

/*!
    @brief Class representing a Tensor.
    @details A tensor is a multi-dimensional array of data.
    This class represents a tensor within the ModularML library. All modularML Tensors are expected to be
    represnted by a underlying 1-D data structure that is row-major. This means that the data is stored in a
    contiguous block of memory with the last dimension changing the fastest. This class provides the basic
    functionality for a tensor including getting and setting elements, reshaping the tensor and checking if
    two tensors are equal.
    @tparam T the type of the data contained in the tensor. E.g. int, float,
    double etc.
*/
template <typename T>
class Tensor {
 public:
  /// @brief The type of the data in the tensor.
  using value_type = T;

  /*!
  @brief Constructor for Tensor class.
  @param shape The shape of the tensor.*/
  explicit Tensor(initializer_list<int> shape)
      : shape(shape),
        offsets(compute_offsets()),
        size(compute_size()) {}

  /*!
  @brief Constructor for Tensor class.
  @param shape The shape of the tensor.*/
  explicit Tensor(array_mml<int> shape)
      : shape(move(shape)),
        offsets(compute_offsets()),
        size(compute_size()) {}

  /*!
  @brief Constructor for creating a tensor with a given shape and all elements initialized to zero.
  @param shape The shape of the tensor.
  */
 Tensor(const std::vector<int>& shape) 
 : data(std::make_shared<Vector_mml<T>>(std::accumulate(shape.begin(), 
  shape.end(), 1, std::multiplies<int>()))),
   shape(shape),
   offsets(compute_offsets()) {}

  /// @brief Move constructor.
  Tensor(Tensor &&other) noexcept
      : shape(array_mml<int>(other.shape)),
        offsets(array_mml<int>(other.offsets)),
        size(other.size) {}

  /// @brief Copy constructor.
  Tensor(const Tensor &other)
      : shape(array_mml<int>(other.shape)),
        offsets(array_mml<int>(other.offsets)),
        size(other.size) {}

  /// @brief Destructor for Tensor class.
  virtual ~Tensor() = default;

  /*!
  @brief Get the shape of the tensor.
  @return A vector of integers representing the shape.*/
  const array_mml<int> &get_shape() const {
    return this->shape;
  }

  /// @brief Get the the total number of elements in the tensor.
  /// @return The total number of elements in the tensor.
  int get_size() const {
    return this->size;
  }

  /// @brief Fills the tensor with a given value.
  /// @param value The value to fill the tensor with.
  void fill(T value) {
    for (uint64_t i = 0; i < this->size; i++) {
      (*this)[i] = value;
    }
  }

  /// @brief Display the tensor.
  /// @return A string representation of the tensor.
  virtual string to_string() const {
    string shp = this->shape.to_string();
    string adr = std::to_string((uintptr_t) this);
    string result = "Tensor: " + adr + " Shape: " + shp;
    return result;
  }

  /*!
  @brief Get the shape as a string.
  @return A string representation of the shape. E.g. [2, 3, 4].*/
  friend ostream &operator<<(ostream &os, const Tensor<T> &tensor) {
    os << tensor.to_string();
    return os;
  }

  /// @brief Get the row-major offsets for the tensor.
  /// @return An array of integers representing the row-major offsets.
  const array_mml<int> &get_offsets() const {
    return this->offsets;
  }

  /*!
  @brief Check if this tensor is not equal to another tensor.
  @param other The tensor to compare with.
  @return True if the tensors are not equal, false otherwise.*/
  bool operator!=(const Tensor<T> &other) const {  // NOSONAR - This is how the function is defined.
    return !(*this == other);
  }

  /*!
  @brief Get an element from the tensor using multi-dimensional indices.
  @param indices A vector of integers representing the indices of the element.
  @return The element at the given indices.*/
  const T &operator[](initializer_list<int> indices) const {
    if (!valid_indices(array_mml<int>(indices))) {
      throw invalid_argument("Invalid Tensor indices");
    }
    const int index = index_with_offset(array_mml<int>(indices));
    return (*this)[index];
  }

  /*!
  @brief Set an element in the tensor using multi-dimensional indices.
  @param indices A vector of integers representing the indices of the element.
  @return The tensor with the element get_mutable_elem.*/
  T &operator[](initializer_list<int> indices) {
    if (!valid_indices(array_mml<int>(indices))) {
      throw invalid_argument("Invalid Tensor indices");
    }
    const int index = index_with_offset(array_mml<int>(indices));
    return (*this)[index];
  }

  /*!
  @brief Get an element from the tensor using multi-dimensional indices.
  @param indices A vector of integers representing the indices of the element.
  @return The element at the given indices.*/
  const T &operator[](array_mml<int> &indices) const {
    if (!valid_indices(indices)) {
      throw invalid_argument("Invalid Tensor indices");
    }
    const int index = index_with_offset(indices);
    return (*this)[index];
  }

  /*!
  @brief Set an element in the tensor using multi-dimensional indices.
  @param indices A vector of integers representing the indices of the element.
  @return The tensor with the element get_mutable_elem.*/
  T &operator[](array_mml<int> &indices) {
    if (!valid_indices(indices)) {
      throw invalid_argument("Invalid Tensor indices");
    }
    const int index = index_with_offset(indices);
    return (*this)[index];
  }

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor.
  void reshape(array_mml<int> &new_shape) {
    if (!valid_shape(new_shape)) {
      throw invalid_argument("Invalid Tensor shape");
    } else {
      this->shape = new_shape;
      this->offsets = compute_offsets();
    }
  }

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor.
  void reshape(initializer_list<int> new_shape) {
    auto new_shape_vec = array_mml<int>(new_shape);
    this->reshape(new_shape_vec);
  }

  /*!
  @brief Move assignment operator.
  @param other The tensor to move.
  @return The moved tensor.*/
  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      *this = move(other);
    }
    return *this;
  }

  /// @brief Check if the tensor is a matrix.
  /// @return True if the tensor is a matrix (has rank 2), false otherwise.
  bool is_matrix() const {
    return this->shape.size() == 2;
  }

  /// @brief Check if the tensor-matrix matches another matrix. Assumes the tensor is a matrix.
  /// @param other The other matrix to compare with.
  /// @return True if the tensor-matrix matches the other matrix, false otherwise.
  bool matrix_match(const Tensor<T> &other) const {
    if (!other.is_matrix()) {
      return false;
    }
    return this->shape[1] == other.shape[0];
  }

  /*!
  @brief Check if this tensor is equal to another tensor.
  @param other The tensor to compare with.
  @return True if the tensors are equal, false otherwise.*/
  bool operator==(const Tensor<T> &other) const {  // NOSONAR - This is how the function is defined.
    if (this->get_shape() != other.get_shape()) {
      return false;
    }

    for (int i = 0; i < this->get_size(); i++) {
      if ((*this)[i] != other[i]) {
        return false;
      }
    }

    return true;
  }

  /*!
  @brief Get an element from the tensor using singel-dimensional index.
  @param index A single integer representing the index of the element.
  @return The element at the given indices.*/
  virtual const T &operator[](int index) const = 0;

  /*!
  @brief Set an element in the tensor using single-dimensional index.
  @param index A single integer representing the index of the element.
  @return The tensor with the element get_mutable_elem.*/
  virtual T &operator[](int index) = 0;

  /// @brief Explicit call to copy the tensor.
  /// @return A shared pointer to the copied tensor.
  virtual shared_ptr<Tensor<T>> copy() const = 0;

 private:
  /// @brief The shape of the tensor.
  array_mml<int> shape;

  /// @brief The row-major offsets for the tensor.
  array_mml<int> offsets;

  /// @brief the size of the tensor.
  uint64_t size;

  /// @brief Row-major offsets for the data structure.
  /// @return a vector of integers representing the offsets.
  array_mml<int> compute_offsets() const {
    const int shape_size = static_cast<int>(shape.size());
    auto computed_offsets = array_mml<int>(shape_size);
    computed_offsets.fill(1);
    for (int i = shape_size - 2; i >= 0; i--) {
      computed_offsets[i] = this->shape[i + 1] * computed_offsets[i + 1];
    }
    return computed_offsets;
  }

  /// @brief Calculate the size of the tensor from the shape.
  /// @return The size of the tensor.
  uint64_t compute_size() const {
    return accumulate(this->shape.begin(), this->shape.end(), 1, multiplies<int>());
  }

  bool valid_shape(const array_mml<int> &new_shape) const {
    return accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>()) == this->get_size();
  }

  /// @brief Check if the indices are valid.
  /// @param indices The indices to check.
  /// @return True if the indices are valid, false otherwise.
  bool valid_indices(const array_mml<int> &indices) const {
    if (indices.size() != this->shape.size()) {
      return false;
    }
    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
      if (indices[i] < 0 || indices[i] >= this->shape[i]) {
        return false;
      }
    }
    return true;
  }

  /// @brief Calculates the 1_D index from the multi-dimensional indices.
  /// @param indices The indices to get the index for.
  /// @return The index.
  int index_with_offset(array_mml<int> indices) const {
    auto index = 0;
    const auto shape_size = static_cast<int>(shape.size());
    for (int i = 0; i < shape_size; i++) {
      index += (indices[i]) * this->offsets[i];
    }
    return index;
  }
};
