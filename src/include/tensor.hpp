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
      @param data Unique pointer to the data structure used to store the tensor data.
      @param am Unique pointer to the arithmetic module used to perform operations on the tensor data.
  */
  Tensor(u_p<DataStructure<T>> data, u_p<ArithmeticModule<T>> am)
      : data(std::move(data)), am(std::move(am)) {}

  /*!
      @brief Destructor for Tensor class.
  */
  ~Tensor() = default;

  /*!
      @brief Get the shape of the tensor.
      @return A vector of integers representing the shape.
  */
  vec<int> get_shape() {
    // implemented here because it's a template class
    return this->data->get_shape();
  }

  /*!
      @brief Get the shape as a string.
      @return A string representation of the shape. E.g. [2, 3, 4].
  */
  std::string get_shape_str() const {
    return this->data->get_shape_str();
  }

  // ARITHMETIC OPERATIONS

  /*!
      @brief Add another tensor to this tensor.
      @param other The tensor to add.
      @return The result of adding the two tensors.
  */
  Tensor<T> operator+(const Tensor<T> &other) const {
    this->am->add(this->data, other.data);
    return *this;
  }

  /*!
      @brief Subtract another tensor from this tensor.
      @param other The tensor to subtract.
      @return The result of subtracting the other tensor from this tensor.
  */
  Tensor<T> operator-(const Tensor<T> &other) const {
    this->am->subtract(this->data, other.data);
    return *this;
  }

  /*!
      @brief Multiply this tensor by another tensor.
      @param other The tensor to multiply by.
      @return The result of multiplying the two tensors.
  */
  Tensor<T> operator*(const Tensor<T> &other) const {
    this->am->multiply(this->data, other.data);
    return *this;
  }

  /*!
      @brief Multiply this tensor by a scalar.
      @param scalar The scalar value to multiply by.
      @return The result of multiplying the tensor by the scalar.
  */
  Tensor<T> operator*(const T &scalar) const {
    this->am->multiply(this->data, scalar);
    return *this;
  }

  /*!
      @brief Divide this tensor by a scalar.
      @param scalar The scalar value to divide by.
      @return The result of dividing the tensor by the scalar.
  */
  Tensor<T> operator/(const T &scalar) const {
    this->am->divide(this->data, scalar);
    return *this;
  }

  /*!
      @brief Check if this tensor is equal to another tensor.
      @param other The tensor to compare with.
      @return True if the tensors are equal, false otherwise.
  */
  bool operator==(const Tensor<T> &other) const {
    return this->am->equal(this->data, other.data);
  }

  /*!
      @brief Check if this tensor is not equal to another tensor.
      @param other The tensor to compare with.
      @return True if the tensors are not equal, false otherwise.
  */
  bool operator!=(const Tensor<T> &other) const {
    return !this->am->equal(this->data, other.data);
  }

 private:
  /// @brief Underlying data structure for the tensor.
  u_p<DataStructure<T>> data;

  /// @brief Underlying arithmetic module for the tensor.
  u_p<ArithmeticModule<T>> am;
};
;
