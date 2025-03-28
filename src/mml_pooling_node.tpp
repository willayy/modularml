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
PoolingNode_mml<T>::PoolingNode_mml(array_mml<uli> kernel_shape,
                                    array_mml<uli> strides,
                                    shared_ptr<Tensor<T>> input,
                                    string auto_pad, uli ceil_mode,
                                    array_mml<uli> dilations,
                                    array_mml<uli> pads)
    : kernel_shape(kernel_shape), strides(strides), input(input),
      dilations(dilations), pads(pads) {
  if (auto_pad != "NOTSET" && auto_pad != "VALID" && auto_pad != "SAME_UPPER" &&
      auto_pad != "SAME_LOWER") {
    throw std::invalid_argument("Invalid padding value! Only 'VALID', "
                                "'SAME_UPPER' and 'SAME_LOWER' are allowed.");
  }
  if (ceil_mode < 0 || ceil_mode > 1) {
    throw std::invalid_argument("Invalid ceil_mode value! Must be 0 or 1");
  }

  this->auto_pad = auto_pad;
  this->ceil_mode = ceil_mode;
  this->output = array_mml<GeneralDataTypes>(2);
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
  if (auto tensor_ptr = std::get_if<std::shared_ptr<Tensor<T>>>(&output[0])) {
    auto &tensor = *tensor_ptr;

    return tensor && tensor->get_size() > 0;

  } else {
    return false;
  }
};

template <typename T>
array_mml<GeneralDataTypes> PoolingNode_mml<T>::getOutputs() const {
  return output;
}

template <typename T> void PoolingNode_mml<T>::forward() {
  array_mml<uli> input_shape = input->get_shape();
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Invalid tensor shape");
  }

  array_mml<uli> output_shape =
      array_mml({input_shape[0], input_shape[1], 1UL, 1UL});

  // Calculate effective kernel size with dilation
  array_mml<uli> effective_kernel_shape =
      array_mml({kernel_shape[0] + (kernel_shape[0] - 1) * (dilations[0] - 1),
                 kernel_shape[1] + (kernel_shape[1] - 1) * (dilations[1] - 1)});

  vector<uli> pad_shape = {pads[0] + pads[1], pads[2] + pads[3]};
  // Calculate output dimensions based on padding type
  for (uli i = 2; i < 4; i++) {
    if (auto_pad == "VALID") {
      if (ceil_mode) {
        output_shape[i] = static_cast<uli>(
            ceil((static_cast<float>(input_shape[i]) -
                  (effective_kernel_shape[i - 2] - 1) * dilations[i - 2]) /
                 static_cast<float>(strides[i - 2])));

      } else {

        output_shape[i] =
            (input_shape[i] -
             (effective_kernel_shape[i - 2] - 1) * dilations[i - 2]) /
                strides[i - 2] +
            1;
      }
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {

      if (ceil_mode) {

        output_shape[i] = static_cast<uli>(
            ceil(static_cast<float>(input_shape[i]) / strides[i - 2]));

      } else {
        output_shape[i] =
            static_cast<uli>(floor((static_cast<float>(input_shape[i]) - 1) /
                                   static_cast<float>(strides[i - 2]))) +
            1;
      }
      pad_shape[i - 2] =
          (output_shape[i] - 1) * strides[i - 2] +
          ((effective_kernel_shape[i - 2] - 1) * dilations[i - 2] + 1) -
          input_shape[i];

    } else {

      if (ceil_mode) {
        output_shape[i] = static_cast<uli>(
            ceil((static_cast<float>(input_shape[i]) + pad_shape[i - 2] -
                  dilations[i - 2] * (effective_kernel_shape[i - 2] - 1) - 1) /
                     strides[i - 2] +
                 1));
      } else {

        output_shape[i] = static_cast<uli>(
            floor((input_shape[i] + pad_shape[i - 2] -
                   dilations[i - 2] * (effective_kernel_shape[i - 2] - 1) - 1) /
                      strides[i - 2] +
                  1));
      }
    }
  }

  pooling(input, input_shape, output_shape, effective_kernel_shape,
          pad_shape[0], pad_shape[1], auto_pad);
}