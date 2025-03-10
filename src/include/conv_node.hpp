#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_tensor.hpp"
#include "mml_gemm.hpp"
#include <string>

/**
 * @class ConvNode
 * @brief A class representing a Convolutional node in a computational graph.
 *
 * This class inherits from the Node class and represents a Conv node
 * in a computational graph.
 */
template <typename T>
class ConvNode : public Node {
    static_assert(
        std::is_same_v<T, double> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, uint>,
        "GemmNode_T supports only double, float, int8");
public:
    using AbstractTensor = Tensor<T>;

    /**
     * @brief Constructor for ConvNode.
     *
     * @param X Shared pointer to the tensor X (data input).
     * @param W Shared pointer to the tensor W (weights).
     * @param B Optional shared pointer to the output tensor (bias).
     * @param Y Shared pointer to the output tensor Y.
     * @param dilations Dilation value along each spatial axis of the filter.
     * @param padding The shape of the convolution kernel.
     * @param kernel_shape The shape of the convolution kernel.
     * @param stride Stride along each spatial axis.
     * @param group number of groups input channels and out channels are divided into.
     */
    ConvNode(shared_ptr<AbstractTensor> X,
             shared_ptr<AbstractTensor> W,
             shared_ptr<AbstractTensor> Y,
             vector<int> dilations,
             vector<int> padding,
             vector<int> kernel_shape,
             vector<int> stride,
             optional<shared_ptr<AbstractTensor>> B = std::nullopt,
             int group = 1)
      : X(X), W(W), B(B), Y(Y),
        dilations(dilations), padding(padding), kernel_shape(kernel_shape), stride(stride) {}

    /**
     * @brief Perform the forward pass convolution computation.
     * 
     * This method performs the forward pass of the convolution operation. It computes the convolution 
     * between the input tensor and the weights (filters), and stores the result in the output tensor.
     * The process is as follows:
     * - Validating input dimensions and checking if inputs are set.
     * - Reshaping the input tensor using im2col (image to column) to transform the convolution into a matrix multiplication.
     * - Flattening the weight tensor.
     * - Performing the matrix multiplication using GEMM (General Matrix Multiply) to compute the convolution result.
     * - Reshaping the result tensor to the appropriate output dimensions.
     * 
     * The output of the convolution is stored in the `Y` tensor.
     * 
     * @throws runtime_error If the input tensor is not correctly shaped or not filled.
     * @throws invalid_argument If any of the padding, stride, or kernel parameters are incorrectly specified.
     * 
     * @see Tensor_mml, im2col, GEMM
     */
    void forward() override {
        validate_inputs();

        // Create a copy of the input
        auto input_copy = X->copy();

        int batch_size = input_copy->get_shape()[0];
        int in_channels = input_copy->get_shape()[1];
        int in_height = input_copy->get_shape()[2];
        int in_width = input_copy->get_shape()[3];

        int kernel_height = W->get_shape()[2];
        int kernel_width = W->get_shape()[3];
        
        int stride_h = stride[0];
        int stride_w = stride[1];
        
        int out_height = (in_height - kernel_height + 2 * padding[0]) / stride_h + 1;
        int out_width = (in_width - kernel_width + 2 * padding[1]) / stride_w + 1;
        
        auto im2col_output = make_shared<Tensor_mml<T>>(std::initializer_list<int>{batch_size * out_height * out_width, in_channels * kernel_height * kernel_width});
        
        im2col(input_copy, im2col_output);
        
        // Flatten the weight tensor (dimensions are Filters x in_channels * kernel_height * kernel_width)
        int out_channels = W->get_shape()[0];
        int flattened_size = in_channels * kernel_height * kernel_width;
        array_mml<int> shapeW({out_channels, flattened_size});
        auto flattened_weights = make_shared<Tensor_mml<T>>(shapeW);

        auto W_tensor = dynamic_pointer_cast<Tensor_mml<T>>(W);

        // TODO Extract into private method
        // Write the values from the weight tensor (W) to the flattened tensor
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int flat_index = ic * kernel_height * kernel_width + kh * kernel_width + kw;
                        (*flattened_weights)[oc * flattened_size + flat_index] =
                            W_tensor->get_data()[oc * in_channels * kernel_height * kernel_width + 
                                        ic * kernel_height * kernel_width + 
                                        kh * kernel_width + kw];
                    }
                }
            }
        }
/* 
        std::cout << "im2col output:" << im2col_output->get_shape() << std::endl;
        for (int i=0; i<im2col_output->get_size(); i++) {
            std::cout << "im2col at index " << i << ": " << im2col_output->get_data()[i] << std::endl;
        }
        std::cout << "flattened weights: " << flattened_weights->get_shape() << std::endl;
        for (int i=0; i<flattened_weights->get_size(); i++) {
            std::cout << "flattened at index " << i << ": " << flattened_weights->get_data()[i] << std::endl;
        } */
        array_mml<int> result_shape({flattened_weights->get_shape()[0], im2col_output->get_shape()[1]});
        auto result_ptr = make_shared<Tensor_mml<T>>(result_shape);

        shared_ptr<GemmModule<T>> gemm = make_shared<Gemm_mml<T>>();
        gemm->gemm_inner_product(
            0, 0,
            flattened_weights->get_shape()[0], im2col_output->get_shape()[1], flattened_weights->get_shape()[1],
            1.0f,
            flattened_weights, flattened_weights->get_shape()[1],
            im2col_output, im2col_output->get_shape()[1],
            0.0f,
            result_ptr, result_ptr->get_shape()[1]);
        
        std::cout << "after gemm shape: " << result_ptr->get_shape() << std::endl;
        for (int i=0; i<result_ptr->get_size(); i++) {
            std::cout << "result at index " << i << ": " << result_ptr->get_data()[i] << std::endl;
        }

        // TODO Make it possible to pass the bias into the gemm call instead
        if (B.has_value()) {
            auto bias = *B;
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int i = 0; i < out_height * out_width; ++i) {
                    result_ptr->get_data()[oc * out_height * out_width + i] += bias->get_data()[oc];
                }
            }
        }

        result_ptr->reshape({batch_size, out_channels, out_height, out_width});

        std::cout << "result after reshape: " << result_ptr->get_shape() << std::endl;
        
        *Y = *result_ptr;
        
    };

    void im2col(shared_ptr<Tensor<T>> input, shared_ptr<Tensor<T>> output) {
        // Get input and output dimensions
        int batch_size = input->get_shape()[0];
        int in_channels = input->get_shape()[1];
        int in_height = input->get_shape()[2];
        int in_width = input->get_shape()[3];

        int kernel_height = W->get_shape()[2];
        int kernel_width = W->get_shape()[3];

        int stride_h = stride[0]; // height stride
        int stride_w = stride[1]; // width stride

        // TODO alter to work for 4 spatial directions
        int padding_h = padding[0]; 
        int padding_w = padding[1]; 

        // Calculate the output height and width based on input size, kernel size, padding, and stride
        int out_height = (in_height - kernel_height + 2 * padding_h) / stride_h + 1;
        int out_width = (in_width - kernel_width + 2 * padding_w) / stride_w + 1;

        // Iterate over each image in the batch
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    int col_index = h * out_width + w; // Column index in im2col matrix

                    for (int c = 0; c < in_channels; ++c) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int input_h = h * stride_h - padding_h + kh;
                                int input_w = w * stride_w - padding_w + kw;

                                int row_index = c * kernel_height * kernel_width + kh * kernel_width + kw;

                                // Compute linear index in output
                                int output_index = row_index * (out_height * out_width) + col_index;

                                if (input_h >= 0 && input_h < in_height && input_w >= 0 && input_w < in_width) {
                                    int input_index = n * (in_channels * in_height * in_width) + c * (in_height * in_width) + input_h * in_width + input_w;
                                    (*output)[output_index] = (*input)[input_index];
                                } else {
                                    (*output)[output_index] = 0; // Padding
                                }
                            }
                        }
                    }
                }
            }
        }        
    }
    
    /**
     * @brief Check if the input(s) are filled.
     * 
     * @return True if the input(s) are filled, false otherwise.
     */
    bool areInputsFilled() const override {
        return X && X->get_size() > 0 &&
               W && W->get_size() > 0 &&
               (!B.has_value() || (B.value() && B.value()->get_size() > 0));
    }

    /**
     * @brief Set the input(s) for the node.
     * 
     * @param inputs The input data to be set, where A is inputs[0], B is inputs[1] and optionally C is inputs[2].
     */
    void setInputs(const array_mml<GeneralDataTypes>& inputs) override {
        if (inputs.size() > 0) {
            auto valueX = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
            *X = *valueX;
        }
            

        if (inputs.size() > 1) {
            auto valueW = std::get<std::shared_ptr<AbstractTensor>>(inputs[1]);
            *W = *valueW;
        }

        if (inputs.size() > 2 && B.has_value()) {
            auto valueB = std::get<std::shared_ptr<AbstractTensor>>(inputs[2]);
            *B.value() = *valueB;
        }
    }

    /**
     * @brief Check if the output(s) are filled.
     * 
     * @return True if the output(s) are filled, false otherwise.
     */
    bool areOutputsFilled() const override {
        return Y && Y->get_size() > 0;
    }

    /**
     * @brief Get the output of the node.
     * 
     * @return The output data.
     */
    array_mml<GeneralDataTypes> getOutputs() const override {
        return array_mml<GeneralDataTypes>{ GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y)) };
    }

private:
    // Inputs
    shared_ptr<AbstractTensor> X; // Input data tensor A has size N x C x H x W.
    shared_ptr<AbstractTensor> W; // The weight tensor used in the convolution.
    optional<shared_ptr<AbstractTensor>> B; // Optional 1D bias tensor.

    // Output
    shared_ptr<AbstractTensor> Y; // Output tensor.

    // Attributes
    vector<int> dilations;
    vector<int> padding;
    vector<int> kernel_shape;
    vector<int> stride;
    int group;

    // Getters for input tensor dimensions
    int get_batch_size() const { return input->get_shape()[0]; }
    int get_in_channels() const { return input->get_shape()[1]; }
    int get_in_height() const { return input->get_shape()[2]; }
    int get_in_width() const { return input->get_shape()[3]; }

    // Weight tensor getters
    int get_kernel_height() const { return W->get_shape()[2]; }
    int get_kernel_width() const { return W->get_shape()[3]; }

    // Getters for the other parameters
    int get_stride_h() const { return stride[0]; }
    int get_stride_w() const { return stride[1]; }
    int get_padding_h() const { return padding[0]; }
    int get_padding_w() const { return padding[1]; }


    void validate_inputs() {
        if (!areInputsFilled())
            throw runtime_error("ConvNode inputs are not fully set.");

        auto x_shape = X->get_shape();
        if (x_shape.size() != 4)
            throw runtime_error("Input tensor must have 4 dimensions: (Features x Channels x Height x Width)");

        if (dilations.size() != 2) {
            throw invalid_argument("Invalid dilations size. Expected a vector of size 2, but got: " + 
                                    std::to_string(dilations.size()) + ".");
        }

        if (padding.size() != 4) {
            throw invalid_argument("Invalid padding vector size. Expected a vector of size 4, but got: " + 
                                    std::to_string(padding.size()) + ".");
        }

        if (kernel_shape.size() != 2) {
            throw invalid_argument("Invalid kernel_shape vector size. Expected a vector of size 2, but got: " + 
                                    std::to_string(kernel_shape.size()) + ".");
        }

        if (stride.size() != 2) {
            throw invalid_argument("Invalid stride vector size. Expected a vector of size 2, but got: " + 
                                    std::to_string(stride.size()) + ".");
        }
    }
    
};