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
  /**
   * @brief Constructor for MaxPool.
   * @param kernel_shape A 2x2 vector of integers representing the kernel
   * shape/pooling window of the layer.
   * @param strides A 2x2 vector of integers representing the strides of the
   * layer.
   * @param input Pointer to the input tensor
   * @param auto_pad (OPTIONAL Parameter representing the padding of the
   * layer. It has a default value of "NOTSET" (no padding) and can also accept
   * "VALID", "SAME_UPPER" and "SAME_LOWER"
   * @param ceiling_mode (OPTIONAL) Whether the output shape should be
   * calcualted with ceil or floor. Accepted values 1 for ceil and 0 for floor.
   * @param dilations (OPTIONAL) Value for dilution of kernel_shape. Default
   * value {1,1}.
   * @param pads (NOT SUPPORTED)
   * @param storage_order (OPTIONAL) Wether the indices of the max values should
   * be calculated as row or column major. 1 is row major and 0 is column major.
   * Defaults to row major.
   */
  MaxPoolingNode_mml(vector<int> kernel_shape, vector<int> strides,
                     shared_ptr<Tensor<T>> input, string auto_pad,
                     int ceiling_mode, vector<int> dilations, vector<int> pads,
                     int storage_order = 0)
      : storage_order(storage_order),
        PoolingNode_mml<T>(kernel_shape, strides, input, auto_pad, ceiling_mode,
                           dilations, pads) {}

private:
  void pooling(const shared_ptr<Tensor<T>> t, array_mml<int> input_shape,
               array_mml<int> output_shape, vector<int> effective_kernel_shape,
               float pad_h, float pad_w, string auto_pad) const override;
  int storage_order;
};

#include "../mml_max_pooling_node.tpp"