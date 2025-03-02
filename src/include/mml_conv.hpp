#pragma once

#include "a_layer.hpp"
#include "globals.hpp"

template <typename T>
class Conv : public Layer<T> {
   public:
    Conv(
        const Tensor<T>& weight,
        const Tensor<T>& bias,
        const Tensor<T>& dilations,
        const int group,
        const Tensor<T>& kernel_shape,
        const vector<int>& padding,
        const vector<int>& stride) : weight(weight),
                                     bias(bias),
                                     dilations(dilations),
                                     group(group),
                                     kernel_shape(kernel_shape),
                                     padding(padding),
                                     stride(stride) {}

    Tensor<T> forward(const Tensor<T>& input) const override;

    Tensor<T> tensor() const override;

    std::unique_ptr<TensorFunction<T>> activation() const override;

   private:
    Tensor<T> weight;  // The weight tensor used in the matrix multiplication
    Tensor<T> bias;
    Tensor<T> dilations;
    int group;
    Tensor<T> kernel_shape;
    vector<int> padding;
    vector<T> stride;
};