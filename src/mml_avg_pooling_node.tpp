#include "mml_avg_pooling_node.hpp"

#include "array_mml.hpp"

template <typename T>
T AvgPoolingNode_mml<T>::pooling(const shared_ptr<Tensor<T>> t,
                                 array_mml<int> shape, int element, int channel,
                                 int in_row_start, int in_col_start,
                                 vector<int> effective_kernel_shape) const {

  // Initialize output tensor with correct dimensions
  output = tensor_mml_p<T>(
      {output_shape[0], output_shape[1], output_shape[2], output_shape[3]});
  output_indices = tensor_mml_p<T>(
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

          T value = 0;
          int k = 0;
          for (int m = 0; m < this->effective_kernel_shape[0];
               m += this->dilations[0]) {
            for (int n = 0; n < this->effective_kernel_shape[1];
                 n += this->dilations[1]) {
              int curr_row = in_row_start + m;
              int curr_col = in_col_start + n;
              // Check if position is within bounds of the original input
              if (curr_row >= 0 && curr_row < shape[2] && curr_col >= 0 &&
                  curr_col < shape[3]) {
                k++;
                value += (*t)[{element, channel, curr_row, curr_col}];
              }
            }
          }
          if (count_include_pad) {
            value = value / (this->kernel_shape[0] * this->kernel_shape[1]);
          }
          value = value / k;

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
  this->output[0] = output;
}