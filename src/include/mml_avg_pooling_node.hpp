#pragma once

#include "mml_pooling_node.hpp"

/**
 * @class AvgPoolingNode_mml
 * @brief Derived class from PoolingNode_mml that performs average pooling.
 * @details This class inherits from the `PoolingNode_mml` base class and
 * implements the specific pooling operation for average pooling. It applies a
 * sliding window over the input tensor and reduces each window to the **average
 * value** of the elements within that window. The output tensor will have the
 * same number of channels as the input tensor but with reduced spatial
 * dimensions (height and width), depending on the stride and padding settings.
 *
 * This class overrides the `pooling()` method to define the behavior of average
 * pooling.
 */
template <typename T> class AvgPoolingNode_mml : public PoolingNode_mml<T> {
public:
  /**
   * @brief Constructor for AvgPool.
   * @param kernel_shape A 2x2 vector of integers representing the kernel
   * shape/pooling window of the layer.
   * @param strides A 2x2 vector of integers representing the strides of the
   * layer.
   * @param input Pointer to the input tensor
   * @param auto_pad (OPTIONAL Parameter representing the padding of the
   * layer. It has a default value of "NOTSET" (no padding) and can also accept
   * "VALID", "SAME_UPPER" and "SAME_LOWER"
   * @param ceil_mode (OPTIONAL) Whether the output shape should be
   * calcualted with ceil or floor. Accepted values 1 for ceil and 0 for floor.
   * @param dilations (OPTIONAL) Value for dilution of kernel_shape. Default
   * value {1,1}.
   * @param pads (NOT SUPPORTED)
   * @param count_include_pad (OPTIONAL) Whether the padding should be included
   * when calculating the edges. 1 for yes and 0 for no. Defaults to no.
   */
  AvgPoolingNode_mml(vector<int> kernel_shape, vector<int> strides,
                     shared_ptr<Tensor<T>> input, string auto_pad = "NOTSET",
                     int ceil_mode = 0, vector<int> dilations = {1, 1},
                     vector<int> pads = {0, 0, 0, 0}, int count_include_pad = 0)
      : count_include_pad(count_include_pad),
        PoolingNode_mml<T>(kernel_shape, strides, input, auto_pad, ceil_mode,
                           dilations, pads) {}

private:
  void pooling(const shared_ptr<Tensor<T>> t, array_mml<int> input_shape,
               array_mml<int> output_shape, vector<int> effective_kernel_shape,
               int pad_h, int pad_w, string auto_pad) override;
  int count_include_pad;
};
#include "../mml_avg_pooling_node.tpp"