#pragma once

#include <modularml>


/**
 * @class mml_TensorFunction_ReLU.hpp
 * @brief A class that implements a tensor function for the tanH function.
 */
class mml_TensorFunction_ReLU : public TensorFunction<float> {
 public:
  /**
   * @brief Apply the ReLU function to the given tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the ReLU function applied to each element.
  */
  Tensor<float> func(const Tensor<float>& t) const {
    auto tensor = t;
    return elementwise_apply(tensor, [](float x) {
      return (x > 0) ? x : 0.0f;
    });
  }

  /**
   * @brief Apply the derivative of the ReLU function to the tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of ReLU applied to each element.
  */
  Tensor<float> derivative(const Tensor<float>& t) const {
    auto tensor = t;
    return elementwise_apply(tensor, [](float x) {
      return (x > 0) ? 1.0f : 0.0f;  // Defaults to 0, like TensorFlow does
    });
  }

  /**
   * @brief Apply the primitive of the ReLU function to the given tensor.

   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the primitive of ReLU applied to each element.
  */
  Tensor<float> primitive(const Tensor<float>& t) const {
    auto tensor = t;
    return elementwise_apply(
        tensor, [](float x) { return (x > 0) ? (x * x) / 2.0f : 0.0f; });
  }
};