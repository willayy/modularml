#pragma once

#include <cmath>
#include <type_traits>

#include "a_tensor.hpp"
#include "globals.hpp"
#include "mml_elementwise.hpp"

/**
 * @class Swish_mml
 * @brief A class that implements a tensor function for the Swish function.
 * @param T The data type of the tensor elements (must be an arithmetic type).
 */
template <typename T>
class Swish_mml : public TensorFunction<T> {
 private:
  mutable mml_elementwise<T> elementwise;  // Determines what version of elementwise to use

 public:
  static_assert(std::is_floating_point<T>::value, "Swish_mml requires a floating-point type (float, double, etc.).");
  /**
  * @brief Apply the Swish function to the given tensor.

  * @param t The tensor to which the function will be applied.
  * @return A new tensor with the Swish function applied to each element.
 */
  shared_ptr<Tensor<T>> func(const shared_ptr<Tensor<T>> t) const {
    return elementwise.apply(t, [](T x) {
      return x / (T(1) + std::exp(-x));  // Compute the Swish function
    });
  }

  /**
   * @brief Apply the derivative of the Swish function to the tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of Swish applied to each element.
  */
  shared_ptr<Tensor<T>> derivative(const shared_ptr<Tensor<T>> t) const {
    return elementwise.apply(t, [](T x) {
      T sigmoid_x = T(1) / (T(1) + std::exp(-x));  // Compute the derivative of the Swish function
      return sigmoid_x + x * sigmoid_x * (T(1) - sigmoid_x);
    });
  }

  /**
   * @brief Apply an approximation of the primitive of the Swish function to the given tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the approximate primitive of Swish applied to each element.
  */
  shared_ptr<Tensor<T>> primitive(const shared_ptr<Tensor<T>> t) const {
    return elementwise.apply(t, [](T x) {
      T sigmoid_x = T(1) / (T(1) + std::exp(-x));            // Compute σ(x)
      return x * sigmoid_x + std::log(T(1) + std::exp(-x));  // Compute integral of Swish
    });
  }
};
