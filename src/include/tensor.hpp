#pragma once

#include <numeric>

#include "a_arithmetic_module.hpp"
#include "a_data_structure.hpp"
#include "globals.hpp"

#define ASSERT_ALLOWED_TYPE(T) static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");

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
      @param am Shared pointer to the arithmetic module used to perform operations on the tensor data.
      @param data Shared pointer to the data structure used to store the tensor data.
      @param shape The shape of the tensor.
  */
  Tensor(unique_ptr<DataStructure<T>> data, unique_ptr<ArithmeticModule<T>> am, vector<int> shape)
      : data(move(data)), am(move(am)), shape(move(shape)), offsets(compute_offsets()) {}

  /*!
      @brief Move constructor.
  */
  Tensor(Tensor &&other) noexcept
      : data(move(other.data)), am(move(other.am)) {}

  /*!
      @brief Copy constructor.
  */
  Tensor(const Tensor &other) {
    this->data = other.data->clone();
    this->am = other.am->clone();
  }

  /*!
      @brief Destructor for Tensor class.
  */
  ~Tensor() = default;

  /*!
      @brief Get the shape of the tensor.
      @return A vector of integers representing the shape.
  */
  const vector<int> get_shape() const {
    return this->shape;
  }

  /*!
      @brief Get the shape as a string.
      @return A string representation of the shape. E.g. [2, 3, 4].
  */
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

  // ARITHMETIC OPERATIONS

  /*!
      @brief Add another tensor to this tensor.
      @param other The tensor to add.
      @return The result of adding the two tensors.
  */
  Tensor<T> operator+(const Tensor<T> &other) const {
    unique_ptr<DataStructure<T>> this_ds_copy = this->data->clone();
    unique_ptr<DataStructure<T>> other_ds_copy = other.data->clone();
    unique_ptr<DataStructure<T>> ds = this->am->add(move(this_ds_copy), move(other_ds_copy));
    unique_ptr<ArithmeticModule<T>> am_copy = this->am->clone();
    auto shape_copy = this->shape;
    return Tensor<T>(move(ds), move(am_copy), shape_copy);
  }

  /*!
      @brief Subtract another tensor from this tensor.
      @param other The tensor to subtract.
      @return The result of subtracting the other tensor from this tensor.
  */
  Tensor<T> operator-(const Tensor<T> &other) const {
    unique_ptr<DataStructure<T>> this_ds_copy = this->data->clone();
    unique_ptr<DataStructure<T>> other_ds_copy = other.data->clone();
    unique_ptr<DataStructure<T>> ds = this->am->subtract(move(this_ds_copy), move(other_ds_copy));
    unique_ptr<ArithmeticModule<T>> am_copy = this->am->clone();
    auto shape_copy = this->shape;
    return Tensor<T>(move(ds), move(am_copy), shape_copy);
  }

  /*!
      @brief Multiply this tensor by another tensor.
      @param other The tensor to multiply by.
      @return The result of multiplying the two tensors.
  */
  Tensor<T> operator*(const Tensor<T> &other) const {
    unique_ptr<DataStructure<T>> this_ds_copy = this->data->clone();
    unique_ptr<DataStructure<T>> other_ds_copy = other.data->clone();
    unique_ptr<DataStructure<T>> ds = this->am->multiply(move(this_ds_copy), move(other_ds_copy));
    unique_ptr<ArithmeticModule<T>> am_copy = this->am->clone();
    auto shape_copy = this->shape;
    return Tensor<T>(move(ds), move(am_copy), shape_copy);
  }

  /*!
      @brief Multiply this tensor by a scalar.
      @param scalar The scalar value to multiply by.
      @return The result of multiplying the tensor by the scalar.
  */
  Tensor<T> operator*(const T &scalar) const {
    unique_ptr<DataStructure<T>> this_ds_copy = this->data->clone();
    unique_ptr<DataStructure<T>> ds = this->am->multiply(move(this_ds_copy), move(scalar));
    unique_ptr<ArithmeticModule<T>> am_copy = this->am->clone();
    auto shape_copy = this->shape;
    return Tensor<T>(move(ds), move(am_copy), shape_copy);
  }

  /*!
      @brief Divide this tensor by a scalar.
      @param scalar The scalar value to divide by.
      @return The result of dividing the tensor by the scalar.
  */
  Tensor<T> operator/(const T &scalar) const {
    unique_ptr<DataStructure<T>> this_ds_copy = this->data->clone();
    unique_ptr<DataStructure<T>> ds = this->am->divide(move(this_ds_copy), move(scalar));
    unique_ptr<ArithmeticModule<T>> am_copy = this->am->clone();
    auto shape_copy = this->shape;
    return Tensor<T>(move(ds), move(am_copy), shape_copy);
  }

  /*!
      @brief Check if this tensor is equal to another tensor.
      @param other The tensor to compare with.
      @return True if the tensors are equal, false otherwise.
  */
  bool operator==(const Tensor<T> &other) const {
    unique_ptr<DataStructure<T>> this_ds_copy = this->data->clone();
    unique_ptr<DataStructure<T>> other_ds_copy = other.data->clone();
    return this->am->equals(move(this_ds_copy), move(other_ds_copy));
  }

  /*!
      @brief Check if this tensor is not equal to another tensor.
      @param other The tensor to compare with.
      @return True if the tensors are not equal, false otherwise.
  */
  bool operator!=(const Tensor<T> &other) const {
    unique_ptr<DataStructure<T>> this_ds_copy = this->data->clone();
    unique_ptr<DataStructure<T>> other_ds_copy = other.data->clone();
    return !this->am->equals(move(this_ds_copy), move(other_ds_copy));
  }

  /*!
      @brief Move assignment operator.
  */
  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      data = move(other.data);
      am = move(other.am);
    }
    return *this;
  }

  /*!
      @brief Get an element from the tensor.
      @param indices A vector of integers representing the indices of the element.
      @return The element at the given indices.
  */
  const T &operator[](vector<int> indices) const {
    if (!valid_indices(indices)) {
      throw std::out_of_range("Invalid Tensor indices");
    } else {
      return this->data->get_elem(index_with_offset(indices));
    }
  }

  /*!
      @brief Set an element in the tensor.
      @param indices A vector of integers representing the indices of the element.
      @return The tensor with the element get_mutable_elem.
  */
  T &operator[](vector<int> indices) {
    if (!valid_indices(indices)) {
      throw std::out_of_range("Invalid Tensor indices");
    } else {
      return this->data->get_mutable_elem(index_with_offset(indices));
    }
  }

 private:
  /// @brief Underlying data structure for the tensor.
  unique_ptr<DataStructure<T>> data;

  /// @brief Underlying arithmetic module for the tensor.
  unique_ptr<ArithmeticModule<T>> am;

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
};
