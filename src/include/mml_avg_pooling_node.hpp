#pragma once

#include "mml_pooling_node.hpp"

/**
 * @class AvgPoolingLayer
 * @brief Derived class from PoolingLayer that performs average pooling.
 * @details This class inherits from the `PoolingLayer` base class and
 * implements the specific pooling operation for average pooling. It applies a
 * sliding window over the input tensor and reduces each window to the **average
 * value** of the elements within that window. The output tensor will have the
 * same number of channels as the input tensor but with reduced spatial
 * dimensions (height and width), depending on the stride and padding settings.
 *
 * This class overrides the `pooling()` method to define the behavior of average
 * pooling.
 */

class AvgPoolingNode_mml : public PoolingNode_mml {
public:
  AvgPoolingNode_mml(vector<int> f, vector<int> s, string p = "valid")
      : PoolingNode_mml<GeneralDataTypes>(f, s, p) {}

private:
  GeneralDataTypes pooling(const shared_ptr<Tensor<GeneralDataTypes>> t,
                           array_mml<int> shape, int element, int channel,
                           int in_row_start, int in_col_start) const override;
};