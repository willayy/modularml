#pragma once

#include "mml_pooling_layer.hpp"

template <typename T>
class AvgPoolingLayer : public PoolingLayer<T> {
 public:
  AvgPoolingLayer(vector<int> f, vector<int> s, string p = "valid")
      : PoolingLayer<T>(f, s, p) {}

 private:
  T pooling(const shared_ptr<Tensor<T>> t, array_mml<int> shape, int element,
            int channel, int in_row_start, int in_col_start) const override;
};

#include "../mml_avg_pooling_layer.tpp"