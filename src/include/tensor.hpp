#pragma once

#include <memory>
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
      @param data Shared pointer to the data structure used to store the tensor data.
      @param am Unique pointer to the arithmetic module used to perform operations on the tensor data.
  */
  Tensor(UPtr<DataStructure<T>> data, UPtr<ArithmeticModule<T>> am)
      : data(move_Ptr(data)), am(move_Ptr(am)) {}

  /*!
      @brief Move constructor.
  */
  Tensor(Tensor &&other) noexcept
      : data(move_Ptr(other.data)), am(move_Ptr(other.am)) {}

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
  Vec<int> get_shape() {
    return this->data->get_shape();
  }

  /*!
      @brief Get the shape as a String.
      @return A String representation of the shape. E.g. [2, 3, 4].
  */
  String get_shape_str() const {
    return this->data->get_shape_str();
  }

  // ARITHMETIC OPERATIONS

  /*!
      @brief Add another tensor to this tensor.
      @param other The tensor to add.
      @return The result of adding the two tensors.
  */
  Tensor<T> operator+(const Tensor<T> &other) const {
    UPtr<DataStructure<T>> this_ds_copy = this->data->clone();
    UPtr<DataStructure<T>> other_ds_copy = other.data->clone();
    UPtr<DataStructure<T>> ds = this->am->add(move_Ptr(this_ds_copy), move_Ptr(other_ds_copy));
    UPtr<ArithmeticModule<T>> am_copy = this->am->clone();
    return Tensor<T>(move_Ptr(ds), move_Ptr(am_copy));
  }

  /*!
      @brief Subtract another tensor from this tensor.
      @param other The tensor to subtract.
      @return The result of subtracting the other tensor from this tensor.
  */
  Tensor<T> operator-(const Tensor<T> &other) const {
    UPtr<DataStructure<T>> this_ds_copy = this->data->clone();
    UPtr<DataStructure<T>> other_ds_copy = other.data->clone();
    UPtr<DataStructure<T>> ds = this->am->subtract(move_Ptr(this_ds_copy), move_Ptr(other_ds_copy));
    UPtr<ArithmeticModule<T>> am_copy = this->am->clone();
    return Tensor<T>(move_Ptr(ds), move_Ptr(am_copy));
  }

  /*!
      @brief Multiply this tensor by another tensor.
      @param other The tensor to multiply by.
      @return The result of multiplying the two tensors.
  */
  Tensor<T> operator*(const Tensor<T> &other) const {
    UPtr<DataStructure<T>> this_ds_copy = this->data->clone();
    UPtr<DataStructure<T>> other_ds_copy = other.data->clone();
    UPtr<DataStructure<T>> ds = this->am->multiply(move_Ptr(this_ds_copy), move_Ptr(other_ds_copy));
    UPtr<ArithmeticModule<T>> am_copy = this->am->clone();
    return Tensor<T>(move_Ptr(ds), move_Ptr(am_copy));
  }

  /*!
      @brief Multiply this tensor by a scalar.
      @param scalar The scalar value to multiply by.
      @return The result of multiplying the tensor by the scalar.
  */
  Tensor<T> operator*(const T &scalar) const {
    UPtr<DataStructure<T>> this_ds_copy = this->data->clone();
    UPtr<DataStructure<T>> ds = this->am->multiply(move_Ptr(this_ds_copy), move_Ptr(scalar));
    UPtr<ArithmeticModule<T>> am_copy = this->am->clone();
    return Tensor<T>(move_Ptr(ds), move_Ptr(am_copy));
  }

  /*!
      @brief Divide this tensor by a scalar.
      @param scalar The scalar value to divide by.
      @return The result of dividing the tensor by the scalar.
  */
  Tensor<T> operator/(const T &scalar) const {
    UPtr<DataStructure<T>> this_ds_copy = this->data->clone();
    UPtr<DataStructure<T>> ds = this->am->divide(move_Ptr(this_ds_copy), move_Ptr(scalar));
    UPtr<ArithmeticModule<T>> am_copy = this->am->clone();
    return Tensor<T>(move_Ptr(ds), move_Ptr(am_copy));
  }

  /*!
      @brief Check if this tensor is equal to another tensor.
      @param other The tensor to compare with.
      @return True if the tensors are equal, false otherwise.
  */
  bool operator==(const Tensor<T> &other) const {
    UPtr<DataStructure<T>> this_ds_copy = this->data->clone();
    UPtr<DataStructure<T>> other_ds_copy = other.data->clone();
    return this->am->equals(move_Ptr(this_ds_copy), move_Ptr(other_ds_copy));
  }

  /*!
      @brief Check if this tensor is not equal to another tensor.
      @param other The tensor to compare with.
      @return True if the tensors are not equal, false otherwise.
  */
  bool operator!=(const Tensor<T> &other) const {
    UPtr<DataStructure<T>> this_ds_copy = this->data->clone();
    UPtr<DataStructure<T>> other_ds_copy = other.data->clone();
    return !this->am->equals(move_Ptr(this_ds_copy), move_Ptr(other_ds_copy));
  }

  /*!
      @brief Move assignment operator.
  */
  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      data = move_Ptr(other.data);
      am = move_Ptr(other.am);
    }
    return *this;
  }

  /*!
      @brief Get an element from the tensor.
      @param indices A vector of integers representing the indices of the element.
      @return The element at the given indices.
  */
  const T &operator[](Vec<int> indices) const {
    return this->data->get_elem(indices);
  }

  /*!
      @brief Set an element in the tensor.
      @param indices A vector of integers representing the indices of the element.
      @return The tensor with the element get_mutable_elem.
  */
  T &operator[](Vec<int> indices) {
    return this->data->get_mutable_elem(indices);
  }

 private:
  /// @brief Underlying data structure for the tensor.
  UPtr<DataStructure<T>> data;

  /// @brief Underlying arithmetic module for the tensor.
  UPtr<ArithmeticModule<T>> am;
};
