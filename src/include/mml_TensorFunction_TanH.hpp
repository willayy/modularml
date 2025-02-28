#pragma once

#include <cmath>
#include "mml_elementwise.hpp"
#include "tensor.hpp"

/**
 * @class mml_TensorFunction_Tanh
 * @brief A class that implements a tensor function for the tanH function.
 */
class tanH_mml : public TensorFunction<float> {
 private:
  mutable mml_elementwise<float> elementwise;  // Determines what version of elementwise to use

 public:
  /**
   * @brief Apply the tanH function to the given tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the tanH function applied to each element.
  */
  Tensor<float> func(const Tensor<float>& t) const {
    return elementwise.apply(t, [](float x) { return std::tanh(x); });
  }

  /**
   * @brief Apply the derivative of the TanH function to the tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of tanH applied to each element.
  */
  Tensor<float> derivative(const Tensor<float>& t) const {
    return elementwise.apply(t, [](float x) {
      float tanh_x = std::tanh(x);  // Compute the derivative of tanh(x)
      return 1.0f - tanh_x * tanh_x;
    });
  }

  /**
   * @brief Apply the primitive of the tanH function to the given tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the primitive of tanH applied to each element.
  */
  Tensor<float> primitive(const Tensor<float>& t) const {
    return elementwise.apply(t, [](float x) {
      return std::log(std::cosh(x));  // Compute the integral of tanh(x)
    });
  }
};