#pragma once

#include "mml_pooling_layer.hpp"

template <typename T>
class AvgPoolingLayer : public PoolingLayer {
 private:
  T pooling(const Tensor<T>& t, vector<int> shape, int element,
            int channel, int in_row_start, int in_col_start) const override;
};