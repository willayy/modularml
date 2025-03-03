#pragma once

#include "mml_pooling_layer.hpp"

template <typename T>
class MaxPoolingLayer : public PoolingLayer<T> {
 public:
  MaxPoolingLayer(vector<int> f, vector<int> s, string p = "valid")
      : PoolingLayer<T>(f, s, p) {}

 private:
  T pooling(const Tensor<T>& t, vector<int> shape, int element,
            int channel, int in_row_start, int in_col_start) const override;
};