#pragma once

#include <cmath>
#include <stdexcept>

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"
#include "mml_pooling_node.hpp"
#include "mml_tensor.hpp"
#include "tuple"

template <typename T>
PoolingNode_mml<T>::PoolingNode_mml(vector<int> kernel_shape,
                                    vector<int> strides,
                                    shared_ptr<Tensor<T>> input,
                                    shared_ptr<Tensor<T>> output,
                                    string auto_pad, int ceiling_mode,
                                    vector<int> dilations, vector<int> pads)
    : kernel_shape(kernel_shape), strides(strides), input(input),
      output(output), ceil_mode(ceiling_mode), dilations(dilations),
      pads(pads) {
  if (auto_pad != "VALID" && auto_pad != "SAME_UPPER" &&
      auto_pad != "SAME_LOWER") {
    throw std::invalid_argument("Invalid padding value! Only 'VALID', "
                                "'SAME_UPPER' and 'SAME_LOWER' are allowed.");
  }
  this->auto_pad = auto_pad;
  this->output = array_mml(2);
};

template <typename T> bool PoolingNode_mml<T>::areInputsFilled() const {
  return input && input->get_size() > 0;
};

template <typename T>
void PoolingNode_mml<T>::setInputs(const array_mml<GeneralDataTypes> &inputs) {
  if (inputs.size() > 0) {
    auto inputValue = std::get<std::shared_ptr<Tensor<T>>>(inputs[0]);
    *input = *inputValue;
  }
};

template <typename T> bool PoolingNode_mml<T>::areOutputsFilled() const {
  return input && input->get_size() > 0;
};

template <typename T>
array_mml<GeneralDataTypes> PoolingNode_mml<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{
      GeneralDataTypes(std::static_pointer_cast<Tensor<T>>(output))};
};

template <typename T> void PoolingNode_mml<T>::forward() {
  array_mml<int> input_shape = input->get_shape();
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Invalid tensor shape");
  }

  vector<int> output_shape = {input_shape[0], input_shape[1], 1, 1};

  // Calculate effective kernel size with dilation
  vector<int> effective_kernel_shape = {
      kernel_shape[0] + (kernel_shape[0] - 1) * (dilations[0] - 1),
      kernel_shape[1] + (kernel_shape[1] - 1) * (dilations[1] - 1)};

  int pad_h = 0;
  int pad_w = 0;

  // Calculate output dimensions based on padding type
  for (int i = 2; i < 4; i++) {
    if (auto_pad == "VALID") {
      if (ceil_mode) {
        output_shape[i] = ceil(
            (input_shape[i] - (effective_kernel_shape[i] - 1) * dilations[i]) /
            strides[i]);
      } else {
        output_shape[i] =
            floor((input_shape[i] -
                   (effective_kernel_shape[i] - 1) * dilations[i]) /
                  strides[i]) +
            1;
      }
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {

      pad_h = static_cast<float>(kernel_shape[0] - 1) / 2;
      pad_w = static_cast<float>(kernel_shape[1] - 1) / 2;

      if (ceil_mode) {
        output_shape[i] = ceil(input_shape[i] / strides[i]);
      } else {
        output_shape[i] = floor((input_shape[i] - 1) / strides[i]) + 1;
      }
    } else {
      throw std::invalid_argument("Unsupported padding mode");
    }
  }

  pooling(input, input_shape, output_shape, effective_kernel_shape, pad_h,
          pad_w, auto_pad);
}