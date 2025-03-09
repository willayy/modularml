#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_tensor.hpp"
#include "mml_gemm.hpp"

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
        std::is_same_v<T, double>   ||
        std::is_same_v<T, float>  ||
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
     * TODO.
     */
    void forward() override {
        if (!areInputsFilled())
            throw runtime_error("ConvNode inputs are not fully set.");

        auto x_shape = X->get_shape();
        if (x_shape.size() != 4)
            throw runtime_error("Input tensor must have 4 dimensions: (Batches x Channels x Height x Width)");

        Tensor<T> input_copy = X->copy();

        //Tensor<T> im2col_matrix = tensor_mml<T>({output_width, output_height, in_channels * kernel_height * kernel_width});
        return;
    };

    void im2col(Tensor<T> input, Tensor<T> output, vector<int> kernel_shape, vector<int> stride) {
        return;
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
};