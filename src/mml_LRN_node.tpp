#pragma once

#include "mml_LRN_node.hpp"

template <typename T>
LRNNode_mml<T>::LRNNode_mml(int size, float alpha, float beta, float bias)
    : size(size), alpha(alpha), beta(beta), bias(bias){};

template <typename T> bool LRNNode_mml<T>::areInputsFilled() const {
  return input && input->get_size() > 0;
};

template <typename T>
void LRNNode_mml<T>::setInputs(const array_mml<GeneralDataTypes> &inputs) {
  if (inputs.size() > 0) {
    auto inputValue = std::get<std::shared_ptr<Tensor<T>>>(inputs[0]);
    *input = *inputValue;
  }
};
template <typename T>
array_mml<GeneralDataTypes> LRNNode_mml<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{
      GeneralDataTypes(std::static_pointer_cast<Tensor<T>>(output))};
};

template <typename T> bool LRNNode_mml<T>::areOutputsFilled() const {
  if (!output)
    return false;
  return output->get_size() > 0;
};

template <typename T> void LRNNode_mml<T>::forward() {

  array_mml<int> shape = input->get_shape();
  output = tensor_mml_p({shape[0], shape[1], shape[2], shape[3]});

  /// Each batch element
  for (int n = 0; n < shape[0]; n++) {
    /// Each channel
    for (int c = 0; c < shape[1]; c++) {
      /// Each row
      for (int h = 0; h < shape[2]; h++) {
        /// Each column
        for (int w = 0; w < shape[3]; w++) {

          /// Region
          int start = max(0, c - (size - 1) / 2);
          int end = min(shape[1] - 1, c + (size - 1) / 2 + ((size - 1) % 2));

          /// Calculate square_sum
          T square_sum = 0;
          for (int i = start; i <= end; i++) {
            square_sum += (*input)[{n, i, h, w}] * (*input)[{n, i, h, w}];
          }
          (*output)[{n, c, h, w}] =
              (*input)[{n, c, h, w}] /
              pow((bias + alpha / size * square_sum), beta);
        }
      }
    }
  }
};