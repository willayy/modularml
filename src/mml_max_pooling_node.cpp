#include "mml_max_pooling_node.hpp"

GeneralDataTypes MaxPoolingNode_mml<GeneralDataTypes>::pooling(
    const shared_ptr < Tensor<GeneralDataTypes> t, array_mml<int> shape,
    int element, int channel, int in_row_start, int in_col_start) const {
  GeneralDataTypes value = std::numeric_limits<T>::lowest();
  for (int m = 0; m < this->filter[0]; m++) {
    for (int n = 0; n < this->filter[1]; n++) {
      int curr_row = in_row_start + m;
      int curr_col = in_col_start + n;
      // Check if position is within bounds of the original input
      if (curr_row >= 0 && curr_row < shape[1] && curr_col >= 0 &&
          curr_col < shape[2]) {
        if (element < 0 || curr_row < 0 || curr_col < 0 || channel < 0 ||
            element >= shape[0] || curr_row >= shape[1] ||
            curr_col >= shape[2] || channel >= shape[3]) {
          throw std::out_of_range("FUCKUP");
        }
        value = std::max(value, (*t)[{element, curr_row, curr_col, channel}]);
      }
    }
  }
  return value;
}