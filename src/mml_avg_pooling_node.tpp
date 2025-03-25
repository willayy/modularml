#include "mml_avg_pooling_node.hpp"

#include "array_mml.hpp"

template <typename T>
void AvgPoolingNode_mml<T>::pooling(const shared_ptr<Tensor<T>> t,
                                    array_mml<uli> input_shape,
                                    array_mml<uli> output_shape,
                                    array_mml<uli> effective_kernel_shape,
                                    uli pad_h, uli pad_w, string auto_pad) {

  // Initialize output tensor with correct dimensions
  shared_ptr<Tensor<T>> output = tensor_mml_p<T>(
      {output_shape[0], output_shape[1], output_shape[2], output_shape[3]});

  // Perform pooling operation
  for (uli element = 0; element < input_shape[0]; element++) {
    for (uli channel = 0; channel < input_shape[1]; channel++) {
      for (uli out_row = 0; out_row < output_shape[2]; out_row++) {
        for (uli out_col = 0; out_col < output_shape[3]; out_col++) {
          uli in_row_start = out_row * this->strides[0];
          uli in_col_start = out_col * this->strides[1];

          // Adjust the starting indices after padding type
          if (auto_pad == "SAME_UPPER") {
            in_row_start -= pad_h / 2;
            in_col_start -= pad_w / 2;
          } else if (auto_pad == "SAME_LOWER") {
            in_row_start -=
                static_cast<uli>(ceil(static_cast<float>(pad_h) / 2));
            in_col_start -=
                static_cast<uli>(ceil(static_cast<float>(pad_w) / 2));
          } else if (auto_pad == "NOTSET") {
            in_row_start -= this->pads[0];
            in_col_start -= this->pads[2];
          }

          T value = 0;
          uli k = 0;
          for (uli m = 0; m < effective_kernel_shape[0];
               m += this->dilations[0]) {
            for (uli n = 0; n < effective_kernel_shape[1];
                 n += this->dilations[1]) {
              uli curr_row = in_row_start + m;
              uli curr_col = in_col_start + n;
              // Check if position is within bounds of the original input
              if (curr_row >= 0 && curr_row < input_shape[2] && curr_col >= 0 &&
                  curr_col < input_shape[3]) {
                k++;
                value += (*t)[{element, channel, curr_row, curr_col}];
              }
            }
          }
          if (count_include_pad) {
            value = value / (this->kernel_shape[0] * this->kernel_shape[1]);
          } else {
            value = value / k;
          }

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