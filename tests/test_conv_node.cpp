#include <gtest/gtest.h>

#include <conv_node.hpp>


TEST(ConvNodeTest, TestConstructor) {
    std::cout << "My test" << std::endl;

    array_mml<int> shapeX({1, 1, 3, 3});
    array_mml<float> X_values({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f});

    array_mml<int> shapeW({1, 1, 2, 2});
    array_mml<float> W_values({
        1.0f, 0.0f, 
        0.0, -1.0f});

    shared_ptr<Tensor_mml<float>> X = make_shared<Tensor_mml<float>>(shapeX, X_values);
    shared_ptr<Tensor_mml<float>> W = make_shared<Tensor_mml<float>>(shapeW, W_values);

    array_mml<int> shapeY({1, 1, 2, 2});
    array_mml<float> Y_values({
        0.0f, 0.0f, 
        0.0f, 0.0f});
    auto Y = make_shared<Tensor_mml<float>>(shapeY, Y_values);

    std::vector<int> dilations = {1, 1};
    std::vector<int> padding = {0, 0, 0, 0};
    std::vector<int> kernel_shape = {2, 2};
    std::vector<int> stride = {1, 1};

    auto B = std::nullopt;

    ConvNode<float> conv(X, W, Y, dilations, padding, kernel_shape, stride, B, 1);

    ASSERT_EQ(Y->get_shape()[0], 1);  // Batch size
    ASSERT_EQ(Y->get_shape()[1], 1);  // Channels
    ASSERT_EQ(Y->get_shape()[2], 2);  // Height
    ASSERT_EQ(Y->get_shape()[3], 2);  // Width
}

TEST(ConvNodeTest, TestImageToColumn) {

    // Define the input tensor shape and values
    array_mml<int> shapeX({1, 1, 3, 3});
    array_mml<float> X_values({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });

    // Define the weight tensor shape and values (for testing im2col only, this might not be used)
    array_mml<int> shapeW({1, 1, 2, 2});
    array_mml<float> W_values({
        1.0f, 0.0f, 
        0.0f, -1.0f
    });

    // Create input and weight tensors
    shared_ptr<Tensor_mml<float>> X = make_shared<Tensor_mml<float>>(shapeX, X_values);
    shared_ptr<Tensor_mml<float>> W = make_shared<Tensor_mml<float>>(shapeW, W_values);

    // Output tensor shape (after applying Conv)
    array_mml<int> shapeY({1, 1, 2, 2});
    array_mml<float> Y_values({
        0.0f, 0.0f, 
        0.0f, 0.0f
    });
    auto Y = make_shared<Tensor_mml<float>>(shapeY, Y_values);

    // Setup other ConvNode parameters
    std::vector<int> dilations = {1, 1};
    std::vector<int> padding = {0, 0, 0, 0};
    std::vector<int> kernel_shape = {2, 2};
    std::vector<int> stride = {1, 1};

    auto B = std::nullopt;

    // Create ConvNode object
    ConvNode<float> conv(X, W, Y, dilations, padding, kernel_shape, stride, B, 1);

    int batch_size = X->get_shape()[0];    // 1
    int in_channels = X->get_shape()[1];   // 1
    int input_height = X->get_shape()[2];  // 3
    int input_width = X->get_shape()[3];   // 3
    int kernel_height = kernel_shape[0];   // 2 
    int kernel_width = kernel_shape[1];    // 2
    int padding_height = padding[0];       // 0
    int padding_width = padding[1];        // 0
    int stride_height = stride[0];         // 1
    int stride_width = stride[1];          // 1

    int output_height = ((input_height - kernel_height + 2 * padding_height) / stride_height) + 1;  // 2
    int output_width = ((input_width - kernel_width + 2 * padding_width) / stride_width) + 1;      // 2

    // Calculate the size of the im2col output
    auto im2col_output = make_shared<Tensor_mml<float>>(std::initializer_list<int>{batch_size * output_height * output_width, in_channels * kernel_height * kernel_width});
        
    // Call im2col method (assuming you have it defined)
    conv.im2col(X, im2col_output);

    // Check the contents of im2col_output to ensure correctness
    std::cout << "Halooooooooooooooooooooo" << std::endl;
}
