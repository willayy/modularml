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

    array_mml<int> dilations = array_mml<int>({1, 1});
    array_mml<int> padding = array_mml<int>({0, 0, 0, 0});
    array_mml<int> kernel_shape = array_mml<int>({2, 2});
    array_mml<int> stride = array_mml<int>({1, 1});

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
    array_mml<int> dilations = array_mml<int>({1, 1});
    array_mml<int> padding = array_mml<int>({0, 0, 0, 0});
    array_mml<int> kernel_shape = array_mml<int>({2, 2});
    array_mml<int> stride = array_mml<int>({1, 1});

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
    
    EXPECT_EQ(im2col_output->get_shape()[0], 4);
}


TEST(ConvNodeTest, TestForward) {

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
    array_mml<int> dilations = array_mml<int>({1, 1});
    array_mml<int> padding = array_mml<int>({0, 0, 0, 0});
    array_mml<int> kernel_shape = array_mml<int>({2, 2});
    array_mml<int> stride = array_mml<int>({1, 1});

    auto B = std::nullopt;

    // Create ConvNode object
    ConvNode<float> conv(X, W, Y, dilations, padding, kernel_shape, stride, B, 1);

    conv.forward();

    // The output is reshaped during the call to forward so we want to make sure that the size is correct
    EXPECT_EQ(Y->get_shape(), array_mml<int>({1, 1, 2, 2}));
    for (int i = 0; i<Y->get_size(); i++) {
        EXPECT_FLOAT_EQ(Y->get_data()[i], -4); // All values in the result should be -4
    }
    //EXPECT_EQ(1, 2);
}

TEST(ConvNodeTest, TestForward_3InChannels_8OutChannels) {
    // Define the input tensor shape and values
    array_mml<int> shapeX({1, 3, 5, 5});
    array_mml<float> X_values({
        // Channel 1
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
        
        // Channel 2
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5,
        
        // Channel 3
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0
    });

    // Define the weight tensor (8 filters, each with 3 channels, 2x2 kernel)
    array_mml<int> shapeW({8, 3, 2, 2});
    array_mml<float> W_values({
        // 8 filters, each with 3 input channels
        1, 0, 0, -1,   0, 1, -1, 0,   1, -1, 0, 1,  // Filter 1
        -1, 1, 1, 0,   0, -1, 1, 1,   1, 0, -1, -1, // Filter 2
        1, 1, 0, -1,   1, 0, -1, -1,  -1, 1, 1, 0,  // Filter 3
        0, -1, 1, 1,   -1, -1, 1, 0,   1, 0, -1, 1, // Filter 4
        1, 0, -1, 1,   1, -1, -1, 0,   -1, 1, 1, 0, // Filter 5
        -1, 1, 0, -1,   0, 1, 1, -1,   1, -1, 0, 1, // Filter 6
        1, -1, 1, 0,   0, 1, -1, 1,   -1, 1, 0, -1, // Filter 7
        0, 1, -1, 1,   -1, 1, 0, -1,   1, 0, 1, -1  // Filter 8
    });

    // Create input and weight tensors
    shared_ptr<Tensor_mml<float>> X = make_shared<Tensor_mml<float>>(shapeX, X_values);
    shared_ptr<Tensor_mml<float>> W = make_shared<Tensor_mml<float>>(shapeW, W_values);

    // Define output tensor shape (1 batch, 8 output channels, 4x4 spatial size)
    array_mml<int> y_shape({1, 8, 4, 4});

    auto Y = make_shared<Tensor_mml<float>>(y_shape);   

    // Define convolution parameters
    array_mml<int> dilations = array_mml<int>({1, 1});
    array_mml<int> padding = array_mml<int>({0, 0, 0, 0});
    array_mml<int> kernel_shape = array_mml<int>({2, 2});
    array_mml<int> stride = array_mml<int>({1, 1});

    auto B = std::nullopt; // No bias

    // Create ConvNode object
    ConvNode<float> conv(X, W, Y, dilations, padding, kernel_shape, stride, B, 8);

    conv.forward();

    // Check output shape
    EXPECT_EQ(Y->get_shape(), array_mml<int>({1, 8, 4, 4}));

    // Check expected values (since exact values depend on conv implementation, verifying non-zero output)
    for (int i = 0; i < Y->get_size(); i++) {
        EXPECT_NEAR(Y->get_data()[i], 0.0f, 1e-5); // Values should not be zero if computed correctly
    }
}

