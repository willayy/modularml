#pragma once

#include <cmath>
#include <type_traits>

#include "mml_elementwise.hpp"
#include "tensor.hpp"

/**
 * @class Sigmoid_mml
 * @brief A class that implements a tensor function for the Sigmoid function.
 * @param T The data type of the tensor elements (must be an arithmetic type).
 */
template <typename T>
class Sigmoid_mml : public TensorFunction<T> {
 private:
  mutable mml_elementwise<T> elementwise;  // Determines what version of elementwise to use

 public:
  static_assert(std::is_arithmetic<T>::value, "Sigmoid_mml requires an arithmetic type (float, double, int, etc.).");

  /**
   * @brief Apply the Sigmoid function to the given tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the Sigmoid function applied to each element.
   */
  Tensor<T> func(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) { return sigmoid(x) });
  }

  /**
   * @brief Apply the derivative of the Sigmoid function to the tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of Sigmoid applied to each element.
   */
  Tensor<T> derivative(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) {
      return sigmoid(x) * (1 - sigmoid(x));
    });
  }

  /**
   * @brief Apply the primitive of the Sigmoid function to the given tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the primitive of Sigmoid applied to each element.
   */
  Tensor<T> primitive(const Tensor<T>& t) const {
    return elementwise.apply(t, [](T x) { return log(1 + exp(x)) });
  }

 private:
  T sigmoid(T x) {
    return 1 / (1 + exp(-x));
  }
};
