#include <gtest/gtest.h>

#include <modularml>

/* Conv(
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
 */

// Test the default constructors
TEST(test_conv, test_constructor) {
    Tensor<int> weights = tensor_mml<int>({1, 1, 3, 3});
    Tensor<int> bias = tensor_mml<int>({1});
    int dilations = 1;
    int group = 1;
    vector<int> padding = {1, 1, 1, 1};
    vector<int> stride = {1, 1};
    
    Conv<int> conv = Conv<int>(weights, bias, dilations, group, padding, stride);

    EXPECT_EQ(conv.get_weight().get_shape(), vector<int>({1, 1, 3, 3}));
    EXPECT_EQ(conv.get_bias().get_shape(), vector<int>({1}));
    EXPECT_EQ(conv.get_dilations(), 1);
    EXPECT_EQ(conv.get_group(), 1);
    EXPECT_EQ(conv.get_padding(), vector<int>({1, 1, 1, 1}));
    EXPECT_EQ(conv.get_stride(), vector<int>({1, 1}));
}

// Test the image to column method
TEST(test_conv, test_image_to_column) {
    Tensor<float> weights = tensor_mml<float>({1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> bias = tensor_mml<float>({1});
    int dilations = 1;
    int group = 1;
    vector<int> padding = {1, 1, 1, 1};
    vector<int> stride = {1, 1};
    
    Conv<float> conv = Conv<float>(weights, bias, dilations, group, padding, stride);

    // Input tensor dimensions (batch size, channels, height, width)
    Tensor<float> input = tensor_mml<float>({1, 1, 3, 3});

    // Fill the input tensor with some known values
    input[0] = 1.0f; input[1] = 2.0f; input[2] = 3.0f; input[3] = 4.0f; input[4] = 5.0f;
    input[5] = 6.0f; input[6] = 7.0f; input[7] = 8.0f; input[8] = 9.0f;

    // Kernel and stride configuration
    int kernel_height = 2;
    int kernel_width = 2;
    int stride_height = 1;
    int stride_width = 1;
    int padding_height = 1;
    int padding_width = 1;

    // Calculate expected output dimensions
    int output_height = (3 + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width = (3 + 2 * padding_width - kernel_width) / stride_width + 1;     


    
    // Call image_to_column function
    Tensor<float> output = conv.image_to_column(input, kernel_height, kernel_width,
                                           stride_height, stride_width,
                                           padding_height, padding_width);

    // Check if the output shape is correct
    EXPECT_EQ(output.get_shape(), vector<int>({output_width, output_height, kernel_height * kernel_width}));
    std::cout << output.get_shape_str() << std::endl;
}

