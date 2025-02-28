#pragma once

#include <numeric>
#include <stdexcept>

#include "a_data_structure.hpp"
#include "globals.hpp"

#define ASSERT_ALLOWED_TYPE_T(T) static_assert(std::is_arithmetic_v<T>, "Tensor must have an arithmetic type.");

/*!
    @brief Class representing a Tensor.
    @details A tensor is a multi-dimensional array of data.
    This class represents a tensor within the ModularML library. ModularML tensors are implemented using a data structure and an
    arithmetic module. The data structure is used to store the data of the tensor and
    the arithmetic module is used to perform arithmetic operations on the data. This allows
    for the implementation of different tensor types with different data structures and arithmetic
    modules.
    @tparam T the type of the data contained in the tensor. E.g. int, float,
    double etc.
    @tparam D the data structure used to store the data. E.g. std::vector,
   std::array, or a custom data structure.
*/
template <typename T>
class Tensor {
 public:
  /*!
  @brief Constructor for Tensor class.
  @param data Shared pointer to the data structure used to store the tensor data.
  @param shape The shape of the tensor.*/
  Tensor(shared_ptr<DataStructure<T>> data, initializer_list<int> shape)
      : data(move(data)),
        shape(vector<int>(shape)),
        offsets(compute_offsets()) {}

  /*!
  @brief Constructor for Tensor class.
  @param data Shared pointer to the data structure used to store the tensor data.
  @param shape The shape of the tensor.*/
  Tensor(shared_ptr<DataStructure<T>> data, vector<int> shape)
      : data(move(data)),
        shape(move(shape)),
        offsets(compute_offsets()) {}

  /// @brief Move constructor.
  Tensor(Tensor &&other) noexcept
      : data(move(other.data)),
        shape(vector<int>(other.shape)),
        offsets(vector<int>(other.offsets)) {}

  /// @brief Copy constructor.
  Tensor(const Tensor &other)
      : data(other.data->clone()),
        shape(vector<int>(other.shape)),
        offsets(vector<int>(other.offsets)) {}

  /// @brief Destructor for Tensor class.
  ~Tensor() = default;

  /*!
  @brief Get the shape of the tensor.
  @return A vector of integers representing the shape.*/
  const vector<int> &get_shape() const {
    return this->shape;
  }

  /// @brief Get the the total number of elements in the tensor.
  /// @return The total number of elements in the tensor.
  int get_size() const {
    return this->data->get_size();
  }

  /*!
  @brief Get the shape as a string.
  @return A string representation of the shape. E.g. [2, 3, 4].*/
  string get_shape_str() const {
    string shape_str = "[";
    for (int i = 0; i < static_cast<int>(this->shape.size()); i++) {
      shape_str += std::to_string(this->shape[i]);
      if (i != static_cast<int>(this->shape.size()) - 1) {
        shape_str += ", ";
      }
    }
    shape_str += "]";
    return shape_str;
  }

  /*!
  @brief Check if this tensor is equal to another tensor.
  @param other The tensor to compare with.
  @return True if the tensors are equal, false otherwise.*/
  bool operator==(const Tensor<T> &other) const {  // NOSONAR - function signature is correct
    if (this->shape != other.shape) {
      return false;
    }
    const auto size = this->data->get_size();
    for (int i = 0; i < size; i++) {
      if ((*this)[i] != other[i]) {
        return false;
      }
    }
    return true;
  }

  /*!
  @brief Check if this tensor is not equal to another tensor.
  @param other The tensor to compare with.
  @return True if the tensors are not equal, false otherwise.*/
  bool operator!=(const Tensor<T> &other) const {  // NOSONAR - function signature is correct
    return !(*this == other);
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

  /*!
  @brief Get an element from the tensor using multi-dimensional indices.
  @param indices A vector of integers representing the indices of the element.
  @return The element at the given indices.*/
  const T &operator[](initializer_list<int> indices) const {
    if (!valid_indices(indices)) {
      throw out_of_range("Invalid Tensor indices");
    } else {
      return this->data->get_elem(index_with_offset(indices));
    }
  }

  /*!
  @brief Set an element in the tensor using multi-dimensional indices.
  @param indices A vector of integers representing the indices of the element.
  @return The tensor with the element get_mutable_elem.*/
  T &operator[](initializer_list<int> indices) {
    if (!valid_indices(indices)) {
      throw out_of_range("Invalid Tensor indices");
    } else {
      return this->data->get_mutable_elem(index_with_offset(indices));
    }
  }

  /*!
  @brief Get an element from the tensor using singel-dimensional index.
  @param index A single integer representing the index of the element.
  @return The element at the given indices.*/
  const T &operator[](int index) const {
    return this->data->get_elem(index);
  }

  /*!
  @brief Set an element in the tensor using single-dimensional index.
  @param index A single integer representing the index of the element.
  @return The tensor with the element get_mutable_elem.*/
  T &operator[](int index) {
    return this->data->get_mutable_elem(index);
  }

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor.
  void reshape(vector<int> &new_shape) {
    if (!valid_shape(new_shape)) {
      throw logic_error("Invalid shape for reshape operation.");
    }
    this->shape = new_shape;
    this->offsets = compute_offsets();
  }

  /// @brief Reshape the tensor.
  /// @param new_shape The new shape of the tensor.
  void reshape(initializer_list<int> new_shape) {
    if (!valid_shape(new_shape)) {
      throw logic_error("Invalid shape for reshape operation.");
    }
    this->shape = vector<int>(new_shape);
    this->offsets = compute_offsets();
  }

 private:
  /// @brief Underlying data structure for the tensor.
  shared_ptr<DataStructure<T>> data;

  /// @brief The shape of the tensor.
  vector<int> shape;

  /// @brief The row-major offsets for the tensor.
  vector<int> offsets;

  /// @brief Check if the indices are valid.
  /// @param indices The indices to check.
  /// @return True if the indices are valid, false otherwise.
  bool valid_indices(const vector<int> &indices) const {
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

  /// @brief Calculates the index of an element in the flat vector containing the data.
  /// @param indices The indices to get the index for.
  /// @return The index.
  int index_with_offset(vector<int> indices) const {
    auto index = 0;
    const auto size = static_cast<int>(shape.size());
    for (int i = 0; i < size; i++) {
      index += (indices[i]) * this->offsets[i];
    }
    return index;
  }

  /// @brief Row-major offsets for the data structure.
  /// @return a vector of integers representing the offsets.
  vector<int> compute_offsets() const {
    const int size = static_cast<int>(shape.size());
    auto computed_offsets = vector<int>(size, 1);
    for (int i = size - 2; i >= 0; i--) {
      computed_offsets[i] = computed_offsets[i + 1] * this->shape[i];
    }
    return computed_offsets;
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
    return this->shape[1] == other.shape[0];
  }

  bool valid_shape(const vector<int> &new_shape) const {
    return accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>()) == this->data->get_size();
  }
};
