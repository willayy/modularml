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
  MaxPoolingNode_mml(vector<int> f, vector<int> s, shared_ptr<Tensor<T>> in,
                     shared_ptr<Tensor<T>> out, string p = "valid")
      : PoolingNode_mml<T>(f, s, in, out, p) {}

private:
  T pooling(const shared_ptr<Tensor<T>> t, array_mml<int> shape, int element,
            int channel, int in_row_start, int in_col_start) const override;
};
#include "../mml_max_pooling_node.tpp"