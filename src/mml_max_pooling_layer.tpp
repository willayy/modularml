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
  vector<T> shape = t.get_shape();
  if (shape.size() != 4) {
    throw invalid_argument("Invalid tensor shape");
  } else {
    int pad_h, pad_w = 0;
    if (padding == "same") {
      pad_h = std::max(0, (filter[0] - 1) / 2);
      pad_w = std::max(0, (filter[1] - 1) / 2);
    }

    // Adjust the input dimensions for padding
    int padded_height = shape[1] + 2 * pad_h;
    int padded_width = shape[2] + 2 * pad_w;

    /// Initialize output tensor with correct dimensions
    shape[1] = std::floor((padded_height - filter[0]) / stride[0]) + 1;
    shape[2] = std::floor((padded_width - filter[1]) / stride[1]) + 1;
    shared_ptr<Tensor<T>> tensor = tensor_mml(shape);

    /// First for loop. For each element in the batch
    for (int i = 0; i < shape[0]; i++) {
      /// Second for loop. For each channel
      for (int j = 0; j < shape[3]; j++) {
        /// Third for loop. Each row in the matrix
        for (int k = 0; k + filter[0] <= padded_height; k += stride[0]) {
          /// Fourth for loop. Each column
          for (int l = 0; l + filter[1] <= padded_width; l += stride[1]) {
            T max_value = t[i][k][l][j];
            for (int m = k; m < k + filter[0]; m++) {
              for (int n = l; n < l + filter[1]; n++) {
                if (m >= pad_h && m < shape[1] + pad_h && n >= pad_w && n < shape[2] + pad_w) {
                  max_value = std::max(max_value, t[i][m - pad_h][n - pad_w][j]);
                }
              }
            }
            // Mapping to the correct output tensor position
            int out_k = k / stride[0];
            int out_l = l / stride[1];
            tensor[i][out_k][out_l][j] = max_value;
          }
        }
      }
    }
    return tensor;
  }
};
