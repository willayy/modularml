#pragma once

#include "a_layer.hpp"

template <typename T>
class MaxPoolingLayer : public Layer {
 public:
  MaxPoolingLayer() {

  };
  Tensor<T> tensor() const override {};

  std::unique_ptr<TensorFunction<T>> activation() const override {};

  Tensor<T> forward(const Tensor<T>& input) const override {};

 private:
};