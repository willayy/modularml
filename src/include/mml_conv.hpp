#pragma once

#include "a_layer.hpp"
#include "globals.hpp"


template <typename T>
class Conv : public Layer<T> {
   public:
    

    Conv(const Tensor<T>& weight, 
         const Tensor<T>& bias, 
         const Tensor<T>& dilations,
         const int group,
         const Tensor<T>& kernel_shape,
         const Tensor<T>& padding, 
         const Tensor<T>& stride);

    
    Tensor<T> forward(const Tensor<T>& input) const override;

    
    Tensor<T> tensor() const override;

    
    std::unique_ptr<TensorFunction<T>> activation() const override;

   private:
    Tensor<T> weight;  // The weight tensor used in the matrix multiplication
    Tensor<T> bias;
    Tensor<T> dilations;
    int group;
    Tensor<T> kernel_shape;
    Tensor<T> padding;
    Tensor<T> stride;
};