#pragma once

#include "mml_elementwise.hpp"
#include "tensor.hpp"
#include <cmath>

/**
 * @class mml_TensorFunction_SoftMax
 * @brief A class that implements a tensor function for the SoftMax function.
 */
class mml_TensorFunction_SoftMax : public TensorFunction<float> {
private:
  mutable mml_elementwise<float>
      elementwise; // Determines what version of elementwise to use

public:
  /**
   * @brief Apply the SoftMax function to the given tensor along the specified
   * axis.
   *
   * @param t The tensor to which the function will be applied.
   * @param axis The axis along which to apply SoftMax. Default is -1 (last
   * axis).
   * @return A new tensor with SoftMax applied.
   */
  Tensor<float> func(const Tensor<float> &t, int axis = -1) const {
    auto shape = t.get_shape();
    if (axis < 0)
      axis += shape.size(); // Adjust for negative axis

    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
      throw std::out_of_range("Invalid axis for SoftMax function");
    }

    Tensor<float> output = t; // Copy input tensor for output
    int axis_size = shape[axis];

    // Compute SoftMax along the specified axis
    for (int i = 0; i < t.get_size() / axis_size; ++i) {
      float max_val = -std::numeric_limits<float>::infinity();
      for (int j = 0; j < axis_size; ++j) {
        max_val = std::max(max_val, t[{i, j}]);
      }

      float sum = 0;
      for (int j = 0; j < axis_size; ++j) {
        output[{i, j}] = std::exp(t[{i, j}] - max_val);
        sum += output[{i, j}];
      }

      for (int j = 0; j < axis_size; ++j) {
        output[{i, j}] /= sum;
      }
    }
    return output;
  }

  /**
   * @brief Apply the derivative of the SoftMax function to the tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the derivative of SoftMax applied.
   */
  Tensor<float> derivative(const Tensor<float> &t) const {
    return elementwise.apply(t, [](float x) { return x * (1.0f - x); });
  }

  /**
   * @brief Apply the primitive of the SoftMax function to the given tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @return A new tensor with the primitive of SoftMax applied.
   */
  Tensor<float> primitive(const Tensor<float> &t) const {
    return elementwise.apply(t, [](float x) { return std::log(x); });
  }
};