#pragma once

#include <cmath>
#include <type_traits>
#include "mml_elementwise.hpp"
#include "tensor.hpp"

/**
 * @class tanH_mml
 * @brief A class that implements a tensor function for the tanh function.
 * @param T The data type of the tensor elements (must be an arithmetic type).
 */
template <typename T>
class tanH_mml : public TensorFunction<T> {
 private:
  mutable mml_elementwise<T> elementwise;  // Determines what version of elementwise to use

 public:
  static_assert(std::is_floating_point<T>::value, "tanH_mml requires a floating-point type (float, double, etc.).");

  /**
   * @brief Apply the tanh function to the given tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the tanh function applied to each element.
   */
  Tensor<T> func(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) { return std::tanh(x); });
  }

  /**
   * @brief Apply the derivative of the tanh function to the tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of tanh applied to each element.
   */
  Tensor<T> derivative(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) {
      T tanh_x = std::tanh(x);  // Compute tanh(x)
      return T(1) - tanh_x * tanh_x;
    });
  }

  /**
   * @brief Apply the primitive of the tanh function to the given tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the primitive of tanh applied to each element.
   */
  Tensor<T> primitive(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) {
      return std::log(std::cosh(x));  // Compute the integral of tanh(x)
    });
  }
};
