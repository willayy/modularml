#pragma once

#include "include/mml_avg_pooling_layer.hpp"

template <typename T>
T AvgPoolingLayer<T>::pooling(const Tensor<T>& t, vector<int> shape, int element,
                              int channel, int in_row_start, int in_col_start) const {
  T value = 0;
  for (int m = 0; m < this->filter[0]; m++) {
    for (int n = 0; n < this->filter[1]; n++) {
      int curr_row = in_row_start + m;
      int curr_col = in_col_start + n;
      // Check if position is within bounds of the original input
      if (curr_row >= 0 && curr_row < shape[1] && curr_col >= 0 && curr_col < shape[2]) {
        value += t[{element, curr_row, curr_col, channel}];
      }
    }
  }
  return value / (this->filter[0] * this->filter[1]);
}