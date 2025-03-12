#include "mml_max_pooling_node.hpp"

template <typename T>
T MaxPoolingNode_mml<T>::pooling(const shared_ptr<Tensor<T>> t,
                                 array_mml<int> shape, int element, int channel,
                                 int in_row_start, int in_col_start,
                                 vector<int> effective_kernel_shape) const {
  T value = std::numeric_limits<T>::lowest();
  for (int m = 0; m < this->effective_kernel_shape[0]; m += this->dilation[0]) {
    for (int n = 0; n < this->effective_kernel_shape[1];
         n += this->dilation[1]) {
      int curr_row = in_row_start + m;
      int curr_col = in_col_start + n;
      // Check if position is within bounds of the original input
      if (curr_row >= 0 && curr_row < shape[2] && curr_col >= 0 &&
          curr_col < shape[3]) {
        if (element < 0 || curr_row < 0 || curr_col < 0 || channel < 0 ||
            element >= shape[0] || curr_row >= shape[2] ||
            curr_col >= shape[3] || channel >= shape[1]) {
          throw std::out_of_range("Out of range");
        }
        value = std::max(value, (*t)[{element, channel, curr_row, curr_col}]);
      }
    }
  }
  return value;
}