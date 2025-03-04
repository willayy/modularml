#pragma once

#include <cmath>
#include <stdexcept>

#include "a_layer.hpp"
#include "string.h"


template <typename T>
class PoolingLayer : public Layer<T> {
 public:
  PoolingLayer(vector<int> f, vector<int> s, string p = "valid") : filter(f), stride(s), padding(p) {};

  Tensor<T> tensor() const override;

  std::unique_ptr<TensorFunction<T>> activation() const override;

  Tensor<T> forward(const Tensor<T>& input) const override;

  virtual T pooling(const Tensor<T>& t, vector<int> shape, int element,
                    int channel, int in_row_start, int in_col_start) const = 0;

 protected:
  vector<int> filter;
  vector<int> stride;
  string padding;
};

#include "../mml_pooling_layer.tpp"