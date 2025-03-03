#pragma once

#include "a_layer.hpp"
#include "string.h"

template <typename T>
class PoolingLayer : public Layer {
 public:
  PoolingLayer(vector<int> f, vector<int> s, string p = "valid") : filter(f), stride(s), padding(p) {};

  Tensor<T> tensor() const override;

  std::unique_ptr<TensorFunction<T>> activation() const override;

  Tensor<T> forward(const Tensor<T>& input) const override;

  virtual T pooling(const Tensor<T>& t, vector<int> shape, int element,
                    int channel, int in_row_start, int in_col_start) const = 0;

 private:
  vector<int> filter;
  vector<int> stride;
  string padding;
};