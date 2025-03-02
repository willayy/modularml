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
        const vector<int>& stride)
        : weight(weight),
          bias(bias),
          dilations(dilations),
          group(group),
          padding(padding),
          stride(stride) {}
    
    /**
     * @brief Performs a forward pass on the input through the convolutional node.
     * 
     * @details See comments within the method for further documentation.
     * 
     * @param input The tensor representing the input. The shape of the tensor HAS to have 4 dimensions: shape = {batches, in_channels, width, height}
     * 
     * @return A tensor object that is the result of the convolution.
     */
    Tensor<T> forward(const Tensor<T>& input) const override {
        // Get all neccesary parameters
        int batch_size = input.get_shape().at(0);     // Number of features (often images)
        int in_channels = input.get_shape().at(1);    // Number of channels per feature
        int in_width = input.get_shape().at(2);       // Width of tensor
        int in_height = input.get_shape().at(3);      // Height of tensor

        int out_channels = weight.get_shape(0);
        int kernel_height = weight.get_shape(2);
        int kernel_width = weight.get_shape(3);
    
        // Prepare the output tensor, final size depends on the kernel size, padding and dilation
        int out_height = (in_height + padding.at(0) + padding.at(1) - dilations * (kernel_height - 1) - 1) / stride[0] + 1;
        int out_width = (in_width + padding.at(2) + padding.at(3) - dilations * (kernel_width - 1) - 1) / stride[1] + 1;

        Tensor<T> result = tensor_mml<T>({out_width, out_height});

        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int y = 0; y < out_height; ++y) {
                    for (int x = 0; x < out_height; ++x) {
                        
                        T sum = bias[oc]; // The bias term that is applied to the entire feature
                        
                        // Now we start sliding the kernel/filter across the input
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int ky = 0; ky < kernel_height; ++ky) {
                                for (int kx = 0; kx < kernel_width; ++kx) {
                                    
                                    int in_x = x * stride.at(0) + kx * dilation - padding.at(0);
                                    int in_y = y * stride.at(1) + ky * dilation - padding.at(1);

                                    // Check that in_x and in_y are within the bounds of the output shape, otherwise continue
                                    if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                                        sum += input[{b, ic, in_y, in_x}] * weight[{oc, ic, ky, kx}];
                                    }
                                }
                            }
                        }
                        result[{b, oc, y, x}] = sum;
                    }   
                }
            }
        }
        return result;
    };  

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
    std::unique_ptr<TensorFunction<T>> activation() const override{
        return nullptr;
    }
   
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

    /// The padding factor defines how many values are added to the input tensor before performing the convolution
    /// This is done to control the spatial dimensions of the output tensor,
    /// Example padding = {1, 1} means that sides left, right and top, bottom repectively are padded with one additional value
    /// Example padding = {1, 1, 2, 2} means that sides left and right have padding = 1, top and bottom have padding = 2
    vector<int> padding;

    /// Defines the how the kernel is moved across the input tensor during the convolution
    vector<int> stride;
};