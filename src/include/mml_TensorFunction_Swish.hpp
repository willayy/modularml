#pragma once

#include <cmath>
#include "mml_elementwise.hpp"
#include "tensor.hpp"

/**
 * @class Swish_mml
 * @brief A class that implements a tensor function for the Swish function.
 */
template <typename T>
class Swish_mml : public TensorFunction<T> {
 private:
  mutable mml_elementwise<float> elementwise;  // Determines what version of elementwise to use

 public:
  /**
  * @brief Apply the Swish function to the given tensor.

  * @param t The tensor to which the function will be applied.
  * @return A new tensor with the Swish function applied to each element.
 */
  Tensor<float> func(const Tensor<float>& t) const {
    return elementwise.apply(t, [](float x) {
      return x / (1.0f + std::exp(-x));  // Compute the Swish function
    });
  }

  /**
   * @brief Apply the derivative of the Swish function to the tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of Swish applied to each element.
  */
  Tensor<float> derivative(const Tensor<float>& t) const {
    return elementwise.apply(t, [](float x) {
      float sigmoid_x = 1.0f / (1.0f + std::exp(-x));  // Compute the derivative of the Swish function
      return sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
    });
  }

  /**
   * @brief Apply an approximation of the primitive of the Swish function to the given tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the approximate primitive of Swish applied to each element.
  */
  Tensor<float> primitive(const Tensor<float>& t) const {
    // This has to be an approximation as there is no known closed-form solution
    // Uses integral approximaation
    return elementwise.apply(t, [](float x) {
      float sigmoid_x = 1.0f / (1.0f + std::exp(-x));        // Compute σ(x)
      return x * sigmoid_x + std::log(1.0f + std::exp(-x));  // Compute integral of Swish
    });
  }
};