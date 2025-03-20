#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_gemm.hpp"
#include "mml_tensor.hpp"

/**
 * @class ConvNode
 * @brief A class representing a Convolutional node in a computational graph.
 *
 * This class inherits from the Node class and represents a Conv node
 * in a computational graph.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
template <typename T>
class ConvNode : public Node {
    static_assert(
        std::is_same_v<T, double> ||
            std::is_same_v<T, float> ||
            std::is_same_v<T, uint>,
        "ConvNode_T only supports double, float, int");

   public:
    using AbstractTensor = Tensor<T>;

    /**
     * @brief Constructor for ConvNode.
     *
     * @param X Shared pointer to the tensor X (data input).
     * @param W Shared pointer to the tensor W (weights).
     * @param Y Shared pointer to the output tensor Y.
     * @param dilations Dilation value along each spatial axis of the filter.
     * @param padding The shape of the convolution kernel.
     * @param kernel_shape The shape of the convolution kernel.
     * @param stride Stride along each spatial axis.
     * @param B Optional shared pointer to the output tensor (bias).
     * @param group number of groups input channels and out channels are divided into.
     */
    ConvNode(shared_ptr<AbstractTensor> X,
             shared_ptr<AbstractTensor> W,
             shared_ptr<AbstractTensor> Y,
             array_mml<int> dilations,
             array_mml<int> padding,
             array_mml<int> kernel_shape,
             array_mml<int> stride,
             optional<shared_ptr<AbstractTensor>> B = std::nullopt,
             int group = 1);

    /**
     * @brief Performs the forward pass convolution operation.
     *
     * The method computes the forward convolution by performing the following steps:
     *
     * 1. **Validate Inputs**: The method first checks that all the inputs are valid. This includes
     *    verifying the dimensions of the input tensor, kernel size, stride, padding, and other
     *    convolution parameters to ensure compatibility.
     *
     * 2. **Apply im2col Transformation**: The input tensor is transformed using the im2col operation.
     *    See explanation below. The transformed data is stored in a temporary tensor, ready
     *    for efficient matrix multiplication.
     *
     * 3. **Perform GEMM (General Matrix Multiply)**: The core of the convolution operation is carried
     *    out using GEMM, which efficiently performs matrix multiplication. The im2col-transformed input
     *    tensor is multiplied with the kernel weights, which are reshaped appropriately. This operation
     *    produces the convolution results in the output tensor, which contains the feature maps after
     *    applying the kernel.
     *
     * 4. **Add Bias (Optional)**: If a bias term is specified, it is added to the output of the GEMM operation.
     *    This bias is applied across the feature maps and is typically used to adjust the activation of the convolutional layer.
     *
     * 5. **Store Result in Output Tensor**: The final result of the convolution operation, after the
     *    optional bias addition, is stored in the output tensor `Y`, which represents the convolved feature maps.
     */
    void forward() override;

    /**
     * @brief Check if the input(s) are filled.
     *
     * @return True if the input(s) are filled, false otherwise.
     */
    bool areInputsFilled() const override;

    /**
     * @brief Set the input(s) for the node.
     *
     * @param inputs The input data to be set.
     */
    void setInputs(const array_mml<GeneralDataTypes>& inputs) override;

    /**
     * @brief Check if the output(s) are filled.
     *
     * @return True if the output(s) are filled, false otherwise.
     */
    bool areOutputsFilled() const override;

    /**
     * @brief Get the output of the node.
     *
     * @return The output data.
     */
    array_mml<GeneralDataTypes> getOutputs() const override;

   private:
    // Inputs
    /**
     * @brief Input data tensor containing the feature map(s) for the convolution.
     *
     * The input tensor typically has the shape [batch_size, in_channels, in_height, in_width].
     * This tensor represents the data that will be convolved with the kernel.
     */
    shared_ptr<AbstractTensor> X;

    /**
     * @brief Weight tensor (kernel) used in the convolution.
     *
     * The kernel tensor typically has the shape [out_channels, in_channels / group, kernel_height, kernel_width]
     * for a grouped convolution. This tensor contains the filters that will be used to convolve the input tensor.
     */
    shared_ptr<AbstractTensor> W;

    /**
     * @brief Optional 1D bias tensor.
     *
     * The bias tensor is added to the output feature map(s) after the convolution.
     * It is typically of shape [out_channels]. If not provided, no bias will be added.
     */
    optional<shared_ptr<AbstractTensor>> B;

    // Output
    /**
     * @brief Output tensor that holds the result of the convolution operation.
     *
     * This tensor typically has the shape [batch_size, out_channels, out_height, out_width],
     * where the output feature map(s) will be stored after performing the convolution.
     */
    shared_ptr<AbstractTensor> Y;

    /**
     * @brief Dilation factors for each dimension of the kernel.
     *
     * Dilation controls the spacing between elements in the kernel. The default is typically [1, 1], meaning no dilation.
     * Dilation increases the receptive field of the kernel without increasing its size.
     */
    array_mml<int> dilations;

    /**
     * @brief Padding to be applied to the input tensor before performing the convolution.
     *
     * Padding for each spatial direction is represented as [top, bottom, left, right].
     * Padding ensures that the convolution can be performed at the borders of the input tensor.
     */
    array_mml<int> padding;

    /**
     * @brief Shape of the kernel (filter).
     *
     * The kernel shape typically has the format [kernel_height, kernel_width].
     * These dimensions determine the size of the region in the input tensor that will be convolved at each step.
     */
    array_mml<int> kernel_shape;

    /**
     * @brief Stride of the convolution operation.
     *
     * Stride specifies the step size for moving the kernel across the input tensor.
     * It is typically represented as [vertical_stride, horizontal_stride].
     */
    array_mml<int> stride;

    /**
     * @brief Number of groups for grouped convolution.
     *
     * If set to 1, a standard convolution is performed. If greater than 1, the input channels are divided into groups,
     * and a grouped convolution is performed. Grouped convolutions can reduce computational complexity.
     */
    int group;

    /**
     * @brief Height of the kernel (filter).
     *
     * Kernel height determines the vertical size of the region in the input tensor to be convolved.
     */
    int kernel_height;

    /**
     * @brief Width of the kernel (filter).
     *
     * Kernel width determines the horizontal size of the region in the input tensor to be convolved.
     */
    int kernel_width;

    /**
     * @brief Number of examples in the batch.
     *
     * The batch size represents how many input tensors will be processed at once.
     */
    int batch_size;

    /**
     * @brief Number of input channels.
     *
     * The input channels correspond to the depth of the input tensor, typically 3 for RGB images.
     */
    int in_channels;

    /**
     * @brief The height of the input tensor.
     *
     * This is the height of the input feature map(s).
     */
    int in_height;

    /**
     * @brief Width of the input tensor.
     *
     * This is the width of the input feature map(s).
     */
    int in_width;

    /**
     * @brief Number of output channels.
     *
     * The number of output channels corresponds to the number of filters used in the convolution.
     */
    int out_channels;

    /**
     * @brief Flips the content of each filter present in the weight kernel.
     *
     * This is done to perform the convolution correctly, otherwise the node would perform a cross-correlation computation
     */
    void flip_kernel();

    /**
     * @brief Performs the im2col transformation on the input tensor.
     *
     * This method extracts patches from the input tensor and flattens them into columns, preparing
     * the data for efficient matrix multiplication in convolution operations. The im2col operation
     * unrolls local patches (based on kernel size, stride, and padding) into column vectors, making
     * convolutions computationally more efficient.
     *
     * @param input A shared pointer to the input tensor, typically of shape
     *              [batch_size, height, width, channels]. This is the data to be transformed into
     *              columns by extracting patches for the convolution.
     *
     * @param output A shared pointer to the output tensor, where the transformed data will be stored.
     *               It will have shape [batch_size, output_height * output_width,
     *               kernel_height * kernel_width * channels], representing the flattened patches
     *               ready for matrix multiplication.
     *
     * @note The im2col operation prepares the input for matrix multiplication with kernel weights
     *       during convolution but does not compute the convolution itself.
     */
    void im2col(shared_ptr<Tensor<T>> input, shared_ptr<Tensor<T>> output);

    /**
     * @brief Performs the addition of the bias to the result.
     *
     * @param result_ptr The tensor to which the bias will be added.
     */
    void add_bias(shared_ptr<Tensor<T>> result_ptr);

    // Getters for input tensor dimensions
    int get_batch_size() const;
    int get_in_channels() const;
    int get_in_height() const;
    int get_in_width() const;

    // Weight tensor getters
    int get_kernel_height() const;
    int get_kernel_width() const;
    int get_out_channels() const;

    // Getters for the other parameters
    int get_stride_height() const;
    int get_stride_width() const;

    // Padding for each spatial direction
    int get_padding_top() const;
    int get_padding_bottom() const;
    int get_padding_left() const;
    int get_padding_right() const;

    // Getter for getting the output height and width
    int get_out_height();
    int get_out_width();

    // Checks the inputs to the convolution node
    void validate_inputs();

    // Updates parameters based on the content of the input and weight tensor
    // This method is executed before forward so that we get the correct parameters.
    void update_parameters();
};

#include "../conv_node.tpp"