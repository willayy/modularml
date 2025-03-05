#pragma once

#include <cmath>
#include <stdexcept>

#include "a_layer.hpp"
#include "string.h"

template <typename T>
class PoolingLayer : public Layer<T> {
 public:
  PoolingLayer(vector<int> f, vector<int> s, string p = "valid") : filter(f), stride(s), padding(p) {};

  shared_ptr<Tensor<T>> tensor() const override;

  std::unique_ptr<TensorFunction<T>> activation() const override;

  shared_ptr<Tensor<T>> forward(const shared_ptr<Tensor<T>> input) const override;

  virtual T pooling(const shared_ptr<Tensor<T>> t, array_mml<int> shape, int element,
                    int channel, int in_row_start, int in_col_start) const = 0;

 protected:
  vector<int> filter;
  vector<int> stride;
  string padding;
};

#include "../mml_pooling_layer.tpp"