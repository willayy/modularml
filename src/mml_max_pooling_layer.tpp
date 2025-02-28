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
    /// Initialize output tensor with correct dimensions
    shape[1] = std::floor((shape[1] - filter[0]) / stride[0]) + 1;
    shape[2] = std::floor((shape[2] - filter[1]) / stride[1]) + 1;
    shared_ptr<Tensor<T>> tensor = tensor_mml(shape);

    /// First for loop. For each element in the batch
    for (int i = 0; i < shape[0]; i++) {
      /// Second for loop. For each channel
      for (int j = 0; j < shape[3]; j++) {
        /// Third for loop. Each row in the matrix
        for (int k = 0; k + filter[0] <= shape[1]; k += stride[0]) {
          /// Fourth for loop. Each column
          for (int l = 0; l + filter[1] <= shape[2]; l += stride[1]) {
            T max_value = t[i][k][l][j];
            for (int m = k; m < k + filter[0]; m++) {
              for (int n = l; n < l + filter[1]; n++) {
                if (max_value < t[i][m][n][j]) {
                  max_value = t[i][m][n][j];
                }
              }
            }
            tensor[i][k][l][j] = max_value;
          }
        }
      }
    }
    return tensor;
  }
};
