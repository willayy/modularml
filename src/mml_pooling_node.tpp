#pragma once

#include <cmath>
#include <stdexcept>

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"
#include "mml_pooling_node.hpp"
#include "mml_tensor.hpp"

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
  if (!output)
    return false;
  return output->get_size() > 0;
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

  // Initialize output tensor with correct dimensions
  output = tensor_mml_p<T>(
      {output_shape[0], output_shape[1], output_shape[2], output_shape[3]});

  // Perform pooling operation
  for (int element = 0; element < input_shape[0]; element++) {
    for (int channel = 0; channel < input_shape[1]; channel++) {
      for (int out_row = 0; out_row < output_shape[2]; out_row++) {
        for (int out_col = 0; out_col < output_shape[3]; out_col++) {
          int in_row_start = out_row * strides[0];
          int in_col_start = out_col * strides[1];

          // Adjust the starting indices after padding type
          if (auto_pad == "SAME_UPPER") {
            in_row_start -= static_cast<int>(std::floor(pad_h));
            in_col_start -= static_cast<int>(std::floor(pad_w));
          } else if (auto_pad == "SAME_LOWER") {
            in_row_start -= static_cast<int>(std::ceil(pad_h));
            in_col_start -= static_cast<int>(std::ceil(pad_w));
          }

          T value = pooling(input, input_shape, element, channel, in_row_start,
                            in_col_start);

          if (element < 0 || out_row < 0 || out_col < 0 || channel < 0 ||
              element >= input_shape[0] || out_row >= output_shape[2] ||
              out_col >= output_shape[3] || channel >= input_shape[1]) {
            throw std::out_of_range("Output tensor indices out of range");
          } else {
            (*output)[{element, channel, out_row, out_col}] = value;
          }
        }
      }
    }
  }
}