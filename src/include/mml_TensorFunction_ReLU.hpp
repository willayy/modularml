#pragma once

#include <type_traits>

#include "mml_elementwise.hpp"
#include "tensor.hpp"

/**
 * @class ReLU_mml
 * @brief A class that implements a tensor function for the ReLU function.
 * @param T The data type of the tensor elements (must be an arithmetic type).
 */
template <typename T>
class ReLU_mml : public TensorFunction<T> {
 private:
  mutable mml_elementwise<T> elementwise;  // Determines what version of elementwise to use

 public:
  static_assert(std::is_arithmetic<T>::value, "ReLU_mml requires an arithmetic type (float, double, int, etc.).");

  /**
   * @brief Apply the ReLU function to the given tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the ReLU function applied to each element.
   */
  Tensor<T> func(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) { return (x > T(0)) ? x : T(0); });
  }

  /**
   * @brief Apply the derivative of the ReLU function to the tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of ReLU applied to each element.
   */
  Tensor<T> derivative(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) {
      return (x > T(0)) ? T(1) : T(0);  // Defaults to 0, like TensorFlow does
    });
  }

  /**
   * @brief Apply the primitive of the ReLU function to the given tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the primitive of ReLU applied to each element.
   */
  Tensor<T> primitive(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) { return (x > T(0)) ? (x * x) / T(2) : T(0); });
  }
};
