#pragma once

#include <cmath>
#include <stdexcept>

#include "include/mml_pooling_layer.hpp"
#include "include/mml_tensors.hpp"

template <typename T>
Tensor<T> PoolingLayer<T>::tensor() const {
  return pooling_size;
};

template <typename T>
std::unique_ptr<TensorFunction<T>> PoolingLayer<T>::activation() const {
  return nullptr;
};

template <typename T>
Tensor<T> PoolingLayer<T>::forward(const Tensor<T>& t) const {
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
    int output_height = (padded_height - filter[0]) / stride[0] + 1;
    int output_width = (padded_width - filter[1]) / stride[1] + 1;

    /// Initialize output tensor with correct dimensions
    vector<int> output_shape = {shape[0], output_height, output_width, shape[3]};
    shared_ptr<Tensor<T>> output_tensor = tensor_mml(output_shape);

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
            T value = pooling(t, shape, element, channel, in_row_start, in_col_start);

            // Store result in output tensor
            (*output_tensor)[element][out_row][out_col][channel] = value;
          }
        }
      }
    }
    /// TODO discuss return type
    return *output_tensor;
  }
}
