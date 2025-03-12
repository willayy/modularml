#pragma once

#include "mml_pooling_node.hpp"

/**
 * @class MaxPoolingNode_mml
 * @brief Derived class from PoolingNode_mml that performs max pooling.
 * @details This class inherits from the `PoolingNode_mml` base class and
 * implements the specific pooling operation for max pooling. It applies a
 * sliding window over the input tensor and reduces each window to the **maximum
 * value** within that window. The output tensor will have the same number of
 * channels as the input tensor but with reduced spatial dimensions (height and
 * width), depending on the stride and padding settings.
 *
 * This class overrides the `pooling()` method to define the behavior of max
 * pooling.
 */
template <typename T> class MaxPoolingNode_mml : public PoolingNode_mml<T> {
public:
  MaxPoolingNode_mml(vector<int> kernel_shape, vector<int> strides,
                     shared_ptr<Tensor<T>> input, shared_ptr<Tensor<T>> output,
                     string auto_pad, int ceiling_mode, vector<int> dilations,
                     vector<int> pads)
      : PoolingNode_mml<T>(kernel_shape, strides, input, output, auto_pad,
                           ceiling_mode, dilations, pads) {}

private:
  T pooling(const shared_ptr<Tensor<T>> t, array_mml<int> shape, int element,
            int channel, int in_row_start, int in_col_start,
            vector<int> effective_kernel_shape) const override;
};
#include "../mml_max_pooling_node.tpp"