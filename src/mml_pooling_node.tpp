#pragma once

#include <cmath>
#include <stdexcept>

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"
#include "mml_pooling_node.hpp"
#include "mml_tensor.hpp"

template <typename T>
PoolingNode_mml<T>::PoolingNode_mml(vector<int> k, vector<int> s,
                                    shared_ptr<Tensor<T>> in,
                                    shared_ptr<Tensor<T>> out, string p)
    : kernel_shape(k), strides(s), input(in), output(out) {
  if (p != "valid" && p != "same") {
    throw std::invalid_argument(
        "Invalid padding value! Only 'valid' and 'same' are allowed.");
  }
  padding = p;
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
  array_mml<int> shape = input->get_shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("Invalid tensor shape");
  } else {
    float pad_h = 0;
    float pad_w = 0;
    if (padding == "same") {
      pad_h = static_cast<float>(kernel_shape[0] - 1) / 2;
      pad_w = static_cast<float>(kernel_shape[1] - 1) / 2;
    }

    // Calculate output dimensions
    int padded_height = shape[1] + static_cast<int>(2 * pad_h);
    int padded_width = shape[2] + static_cast<int>(2 * pad_w);
    int output_height = (padded_height - kernel_shape[0]) / strides[0] + 1;
    int output_width = (padded_width - kernel_shape[1]) / strides[1] + 1;

    // Remove comment for debug
    //  std::cerr << "Padded height: " << padded_height << " Padded width: " <<
    //  padded_width << "\n";

    /// Initialize output tensor with correct dimensions
    output = tensor_mml_p<T>({shape[0], output_height, output_width, shape[3]});

    /// First for loop. For each element in the batch
    for (int element = 0; element < shape[0]; element++) {
      /// Second for loop. For each channel
      for (int channel = 0; channel < shape[3]; channel++) {
        /// Third for loop. Each row in the output matrix
        for (int out_row = 0; out_row < output_height; out_row++) {
          /// Fourth for loop. Each output column
          for (int out_col = 0; out_col < output_width; out_col++) {
            // Calculate input region start (with stride)
            int in_row_start =
                out_row * strides[0] - static_cast<int>(std::floor(pad_h));
            int in_col_start =
                out_col * strides[1] - static_cast<int>(std::floor(pad_w));
            /// Initialize lowest value for type T
            T value = pooling(input, shape, element, channel, in_row_start,
                              in_col_start);

            // Store result in output tensor
            if (element < 0 || out_row < 0 || out_col < 0 || channel < 0 ||
                element >= shape[0] || out_row >= output_height ||
                out_col >= output_width || channel >= shape[3]) {
              throw std::out_of_range("Output tensor indices out of range");
            } else {
              (*output)[{element, out_row, out_col, channel}] = value;
            }
          }
        }
      }
    }
    /// Remove comment for debug
    // std::cerr << "Resulting tensor: " << (*output_tensor).to_string() <<
    // "\n";
  }
}
