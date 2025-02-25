#pragma once

#include "a_layer.hpp"
#include "a_tensor_func.hpp"
#include "tensor.hpp"

template <typename T>
class LinearLayer : public Layer<T> {
 private:
  Tensor<T> weights;
  Tensor<T> biases;

 public:
  LinearLayer(const Tensor<T>& w, const Tensor<T>& b);

  Tensor<T> tensor() const override;

  std::unique_ptr<TensorFunction<T>> activation() const override;

  Tensor<T> forward(const Tensor<T>& input) const override;
};