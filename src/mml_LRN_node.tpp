#pragma once

#include "mml_LRN_node.hpp"

template <typename T>
LRNNode_mml<T>::LRNNode_mml(shared_ptr<Tensor<T>> input, int size, float alpha,
                            float beta, float bias)
    : input(input), size(size), alpha(alpha), beta(beta), bias(bias){};

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

  array_mml<uli> shape = input->get_shape();
  output = tensor_mml_p<T>({shape[0], shape[1], shape[2], shape[3]});

  /// Each batch element
  for (uli n = 0; n < shape[0]; n++) {
    /// Each channel
    for (uli c = 0; c < shape[1]; c++) {
      /// Each row
      for (uli h = 0; h < shape[2]; h++) {
        /// Each column
        for (uli w = 0; w < shape[3]; w++) {

          /// Region
          uli start = std::max(0UL, c - (size - 1) / 2);
          uli end =
              std::min(shape[1] - 1, c + (size - 1) / 2 + ((size - 1) % 2));

          /// Calculate square_sum
          T square_sum = 0;
          for (uli i = start; i <= end; i++) {
            square_sum += (*input)[{n, i, h, w}] * (*input)[{n, i, h, w}];
          }
          (*output)[{n, c, h, w}] =
              (*input)[{n, c, h, w}] /
              std::pow((bias + alpha / size * square_sum), beta);
        }
      }
    }
  }
};