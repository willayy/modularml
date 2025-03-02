#pragma once

#include "a_layer.hpp"
#include "globals.hpp"

template <typename T>
class Conv : public Layer<T> {
   public:
    Conv(
        const Tensor<T>& weight,
        const Tensor<T>& bias,
        int dilations = 1,
        const int group = 1,
        const vector<int>& kernel_shape,
        const vector<int>& padding,
        const vector<int>& stride) : weight(weight),
                                     bias(bias),
                                     dilations(dilations),
                                     group(group),
                                     kernel_shape(kernel_shape),
                                     padding(padding),
                                     stride(stride) {}

    Tensor<T> forward(const Tensor<T>& input) const override;


    /**
     * @brief Returns the weight tensor used in the instance.
     *
     * This function returns the weight tensor. It is a placeholder for now.
     *
     * @return A tensor object representing the weights.
     */
    Tensor<T> tensor() const override {
        return weight;
    };

    
    /**
     * @brief Returns the activation function for the instance.
     *
     * This function is used for layers with activation functions. For Conv, it is not implemented.
     *
     * @return A nullptr since Conv does not have an activation function.
     */
    std::unique_ptr<TensorFunction<T>> activation() const override {
        return nullptr
    };

   private:
    /// The weight tensor used in the matrix multiplication.
    /// This tensor represents the weights (filters) used in the convolutional operation.
    /// It has a shape determined by the kernel dimensions and the number of output channels.
    /// The shape of the weight tensor being = {64, 3, 11, 11} means:
    /// out_channels = 64
    /// in_channels = 3
    /// kernel height = 11
    /// kernel width = 11
    Tensor<T> weight;  
    
    /// The bias tensor used in the convolutional operation.
    /// This tensor adds a bias term to the output after the convolution. 
    /// It typically has one value per output channel, applying the a value uniformly for each slice in the output tensor
    Tensor<T> bias; 

    /// The dilations tensor defines the spacing between kernel elements in the convolution
    /// It adds dilation - 1 amount of space between the elements of the kernel.
    /// 3x3 Kernel with dilation = 1 effectively makes the kernel 5x5
    int dilations;

    /// The group determines the number of groups in the convolution.
    /// A value of 1 means a standard convolution, a value greater than 1 is used in grouped convolutions
    int group;

    /// The kernel_shape tensor defines the dimensions of the kernel used in the convolution
    /// This can be a 2D or 3D shape depending on the operation
    vector<int> kernel_shape;

    /// The padding factor defines how many values are added to the input tensor before performing the convolution
    /// This is done to control the spatial dimensions of the output tensor,
    /// Example padding = {1, 1} means that sides left, right and top, bottom repectively are padded with one additional value
    /// Example padding = {1, 1, 2, 2} means that sides left and right have padding = 1, top and bottom have padding = 2
    vector<int> padding;

    /// Defines the how the kernel is moved across the input tensor during the convolution
    vector<int> stride;
};