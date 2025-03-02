#pragma once

#include <cmath>
#include <stdexcept>

#include "include/mml_max_pooling_layer.hpp"
#include "include/mml_tensors.hpp"

template <typename T>
Tensor<T> MaxPoolingLayer<T>::tensor() const {
  return pooling_size;
};

template <typename T>
std::unique_ptr<TensorFunction<T>> MaxPoolingLayer<T>::activation() const {
  return nullptr;
};

template <typename T>
Tensor<T> MaxPoolingLayer<T>::forward(const Tensor<T>& t) const {
  vector<int> shape = t.get_shape();
  if (shape.size() != 4) {
    throw invalid_argument("Invalid tensor shape");
  } else {
    int pad_h = 0;
    int pad_w = 0;
    if (padding == "same") {
      pad_h = std::max(0, (filter[0] - 1) / 2);
      pad_w = std::max(0, (filter[1] - 1) / 2);
    }

    // Calculate output dimensions
    int padded_height = shape[1] + 2 * pad_h;
    int padded_width = shape[2] + 2 * pad_w;
    int output_height = std::floor((padded_height - filter[0]) / stride[0]) + 1;
    int output_width = std::floor((padded_width - filter[1]) / stride[1]) + 1;

    /// Initialize output tensor with correct dimensions
    vector<int> output_shape = {shape[0], output_height, output_width, shape[3]};
    Tensor<T> output_tensor = *(tensor_mml(output_shape));

    /// First for loop. For each element in the batch
    for (int element = 0; element < shape[0]; element++) {
      /// Second for loop. For each channel
      for (int channel = 0; channel < shape[3]; channel++) {
        /// Third for loop. Each row in the output matrix
        for (int out_row = 0; out_row < output_height; out_row++) {
          /// Fourth for loop. Each output column
          for (int out_col = 0; out_col < output_width; out_col++) {
            // Calculate input region start (with stride)
            int in_row_start = out_row * stride[0] - pad_h;
            int in_col_start = out_col * stride[1] - pad_w;
            /// Initialize lowest value for type T
            T max_value = std::numeric_limits<T>::lowest();
            for (int m = 0; m < filter[0]; m++) {
              for (int n = 0; n < filter[1]; n++) {
                int curr_row = in_row_start + m;
                int curr_col = in_col_start + n;
                // Check if position is within bounds of the original input
                if (curr_row >= 0 && curr_row < shape[1] && curr_col >= 0 && curr_col < shape[2]) {
                  max_value = std::max(max_value, t[element][curr_row][curr_col][channel]);
                }
              }
            }

            // Store result in output tensor
            output_tensor[element][out_row][out_col][channel] = max_value;
          }
        }
      }
    }
    return output_tensor;
  }
}
