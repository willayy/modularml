#include "mml_max_pooling_node.hpp"
#include "tuple"

template <typename T>
void MaxPoolingNode_mml<T>::pooling(const shared_ptr<Tensor<T>> t,
                                    array_mml<int> shape, int element,
                                    int channel, int in_row_start,
                                    int in_col_start,
                                    vector<int> effective_kernel_shape) const {

  tuple<T, int> result;

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

          T value = std::numeric_limits<T>::lowest();
          int index = 0;
          for (int m = 0; m < this->effective_kernel_shape[0];
               m += this->dilation[0]) {
            for (int n = 0; n < this->effective_kernel_shape[1];
                 n += this->dilation[1]) {
              int curr_row = in_row_start + m;
              int curr_col = in_col_start + n;
              // Check if position is within bounds of the original input
              if (curr_row >= 0 && curr_row < shape[2] && curr_col >= 0 &&
                  curr_col < shape[3]) {
                if (element < 0 || curr_row < 0 || curr_col < 0 ||
                    channel < 0 || element >= shape[0] ||
                    curr_row >= shape[2] || curr_col >= shape[3] ||
                    channel >= shape[1]) {
                  throw std::out_of_range("Out of range");
                }
                if ((*t)[{element, channel, curr_row, curr_col}] > value) {
                  value = (*t)[{element, channel, curr_row, curr_col}];
                  if (storage_order) {
                    index = curr_col * input_shape[2] + curr_row;
                  } else {
                    index = curr_row * input_shape[3] + curr_col;
                  }
                }
                value = std::max(value,
                                 (*t)[{element, channel, curr_row, curr_col}]);
              }
            }
          }

          tuple<T, int> result = make_tuple(value, index)

              if (element < 0 || out_row < 0 || out_col < 0 || channel < 0 ||
                  element >= input_shape[0] || out_row >= output_shape[2] ||
                  out_col >= output_shape[3] || channel >= input_shape[1]) {
            throw std::out_of_range("Output tensor indices out of range");
          }
          else {
            (*output)[{element, channel, out_row, out_col}] = get<0>(result);
            (*output_indices)[{element, channel, out_row, out_col}] =
                get<1>(result);
          }
        }
      }
    }
  }

  this->output[0] = output;
  this->output[1] = output_indices;
}
