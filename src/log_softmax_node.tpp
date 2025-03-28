#pragma once

#include "log_softmax_node.hpp"

template <typename T>
LogSoftMaxNode<T>::LogSoftMaxNode(shared_ptr<const AbstractTensor> X,
                                  shared_ptr<AbstractTensor> Y, uli axis)
    : X(X), Y(Y), axis(axis) {}

template <typename T> void LogSoftMaxNode<T>::forward() {
  if (!areInputsFilled())
    throw runtime_error("LogSoftMaxNode inputs are not fully set.");

  if (!Y)
    throw runtime_error("Output tensor Y is not allocated.");

  // If axis is negative
  if (((int) axis) < 0)
    axis += X->get_shape().size();

  if (axis >= X->get_shape().size())
    throw runtime_error("Invalid axis: " + std::to_string(axis));

  // Currently this only supports input tensors that are 2D, this is the most
  // common shape. Currently there is no general solution until we can slice the
  // tensor and retreive axi from the tensor

  auto input_copy = X->copy();

  // For each batch, in the input
  for (uli b = 0; b < input_copy->get_shape()[0]; b++) {
    // Find the maximum value in the row for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (uli c = 0; c < input_copy->get_shape()[axis]; c++) {
      max_val = std::max(max_val, (*input_copy)[{b, c}]);
    }

    // Exponentiate and accumulate the sum
    float sum = 0;
    std::vector<float> exp_values(input_copy->get_shape()[axis]);
    for (uli c = 0; c < input_copy->get_shape()[axis]; c++) {
      float value = (*input_copy)[{b, c}] - max_val;
      exp_values[c] = std::exp(value);
      sum += exp_values[c];
    }

    for (uli c = 0; c < input_copy->get_shape()[axis]; c++) {
      // Apply soft max and perform log on the result
      (*input_copy)[{b, c}] = std::log(exp_values[c] / sum);
    }
  }

  *Y = *input_copy;
}

template <typename T> bool LogSoftMaxNode<T>::areInputsFilled() const {
  return X && X->get_size() > 0;
}

template <typename T>
void LogSoftMaxNode<T>::setInputs(const array_mml<GeneralDataTypes> &inputs) {
  if (inputs.size() < 1)
    throw runtime_error("LogSoftMaxNode expects at least one input: X.");
  auto valueX = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);

  if (!X || !valueX)
    throw runtime_error("Failed to cast X or input X to Tensor<T>.");
  X = *valueX;
}

template <typename T> bool LogSoftMaxNode<T>::areOutputsFilled() const {
  if (!Y)
    return false;
  return Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> LogSoftMaxNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{
      GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}