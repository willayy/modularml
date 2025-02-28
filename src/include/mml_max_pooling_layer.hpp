#pragma once

#include "a_layer.hpp"

template <typename T>
class MaxPoolingLayer : public Layer {
 public:
  MaxPoolingLayer(vector<int> p, vector<int> s) : filter(p), stride(s) {};

  Tensor<T> tensor() const override {};

  std::unique_ptr<TensorFunction<T>> activation() const override {};

  Tensor<T> forward(const Tensor<T>& input) const override {};

 private:
  vector<int> filter;
  vector<int> stride;
};