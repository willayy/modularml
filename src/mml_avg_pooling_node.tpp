#include "mml_avg_pooling_node.hpp"

#include "array_mml.hpp"

template <typename T>
T AvgPoolingNode_mml<T>::pooling(const shared_ptr<Tensor<T>> t,
                                 array_mml<int> shape, int element, int channel,
                                 int in_row_start, int in_col_start) const {
  T value = 0;
  for (int m = 0; m < this->kernel_shape[0]; m++) {
    for (int n = 0; n < this->kernel_shape[1]; n++) {
      int curr_row = in_row_start + m;
      int curr_col = in_col_start + n;
      // Check if position is within bounds of the original input
      if (curr_row >= 0 && curr_row < shape[1] && curr_col >= 0 &&
          curr_col < shape[2]) {
        value += (*t)[{element, curr_row, curr_col, channel}];
      }
    }
  }
  return value / (this->kernel_shape[0] * this->kernel_shape[1]);
}