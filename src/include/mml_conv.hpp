#pragma once

#include "a_layer.hpp"
#include "globals.hpp"

template <typename T>
class Conv : public Layer<T> {
   public:
    Conv(
        const Tensor<T>& weight,
        const Tensor<T>& bias,
        int dilations,
        const int group,
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
        int batch_size = input.get_shape().at(0);   // Number of features (often images)
        int in_channels = input.get_shape().at(1);  // Number of channels per feature
        int in_width = input.get_shape().at(2);     // Width of tensor
        int in_height = input.get_shape().at(3);    // Height of tensor

        int out_channels = weight.get_shape(0);
        int kernel_height = weight.get_shape(2);
        int kernel_width = weight.get_shape(3);

        // Prepare the output tensor, final size depends on the kernel size, padding and dilation
        int out_height = (in_height + padding.at(0) + padding.at(1) - dilations * (kernel_height - 1) - 1) / stride[0] + 1;
        int out_width = (in_width + padding.at(2) + padding.at(3) - dilations * (kernel_width - 1) - 1) / stride[1] + 1;

        // Prepare the image_to_column matrix, (tensor still tho hehe)
        Tensor<T> im2col_matrix = tensor_mml<T>({kernel_height * kernel_width * in_channels, out_height * out_width});

        image_to_column(input, im2col_matrix, kernel_height, kernel_width, stride[0], stride[1], padding[0], padding[1]);


        return nullptr;
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
    std::unique_ptr<TensorFunction<T>> activation() const override {
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

    // To utilize GEMM we use im2col operation to convert the input 4D, into a 2D matrix
    Tensor<T> image_to_column(const Tensor<T>& input,
                              int kernel_height, int kernel_width,
                              int stride_height, int stride_width,
                              int padding_height, int padding_width) {
        int batches = input.get_shape().at(0);
        int in_channels = input.get_shape().at(1);  // Number of channels per feature
        int in_width = input.get_shape().at(2);     // Width of input tensor
        int in_height = input.get_shape().at(3);    // Height of input tensor

        int output_height = (in_height + 2 * padding_height - kernel_height) / stride_height + 1;
        int output_width = (in_width + 2 * padding_width - kernel_width) / stride_width + 1;
        
        int output_size = batches * in_channels * kernel_height * kernel_width * output_height * output_width;

        Tensor<T> output = tensor_mml<T>({output_width, output_height, in_channels * kernel_height * kernel_width});

        int col_index = 0;
        for (int b = 0; b < batches; ++b)
        for (int c = 0; c < in_channels; ++c) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    for (int h = 0; h < output_height; ++h) {
                        for (int w = 0; w < output_width; ++w) {
                            int in_h = h * stride_height + kh - padding_height;
                            int in_w = w * stride_width + kw - padding_width;
                            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                output[col_index++] = input[(b * in_channels * in_height * in_width) + 
                                    (c * in_height * in_width) + 
                                    (in_h * in_width) + in_w];
                            } else {
                                output[col_index++] = 0; // Zero padding if out of bounds
                            }
                        }
                    }
                }
            }
        }
        return output;
    } 
};