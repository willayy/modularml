#pragma once

#include <cmath>
#include "a_tensor_func.hpp"
#include "mml_elementwise.hpp"

/**
 * @class mml_TensorFunction_Tanh
 * @brief A class that implements a tensor function for the tanH function.
 */
class mml_TensorFunction_Tanh : public TensorFunction<float> {
    public:

    /**
     * @brief Apply the tanH function to the given tensor.
 
     * @param t The tensor to which the function will be applied.
     * @return A new tensor with the tanH function applied to each element.
    */
    Tensor<float> func(const Tensor<float>& t) const{
        auto tensor = t;
        return elementwise_apply(tensor, [](float x) {
            return std::tanh(x);
        });
        return tensor;
    }
    /**
     * @brief Apply the derivative of the TanH function to the tensor.
 
     * @param t The tensor to which the function will be applied.
     * @return A new tensor with the derovative of tanH applied to each element.
    */
    Tensor<float> derivative(const Tensor<float>& t) const{
        auto tensor = t;
        return elementwise_apply(tensor, [](float x) {
            float tanh_x = std::tanh(x);  // Compute the derivative of tanh(x)
            return 1.0f - tanh_x * tanh_x;
        });
    }
    /**
     * @brief Apply the primitive of the tanH function to the given tensor.
 
     * @param t The tensor to which the function will be applied.
     * @return A new tensor with the primitive of tanH applied to each element.
    */    
    Tensor<float> primitive(const Tensor<float>& t) const{
        auto tensor = t;
        return elementwise_apply(tensor, [](float x) {
            return std::log(std::cosh(x));  // Compute the integral of tanh(x)
        });
    }
};