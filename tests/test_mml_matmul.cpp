#include <gtest/gtest.h>

#include <modularml>

TEST(test_MatMul, test_constructor) {
    Tensor<float> weights = tensor_mml<float>({2, 2}, {0.1, 0.2, 0.3, 0.4});
    MatMul<float> matmul = MatMul<float>(weights);
    
    EXPECT_EQ(matmul.get_weights().get_shape(), vector<int>({2, 2}));
}

TEST(test_MatMul, test_forward_2d) {
    Tensor<float> input = tensor_mml<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> weight = tensor_mml<float>({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    MatMul<float> matmul(weight);

    Tensor<float> result = matmul.forward(input);
    
    EXPECT_FLOAT_EQ(result[0], 19.0f);
    EXPECT_FLOAT_EQ(result[1], 22.0f);
    EXPECT_FLOAT_EQ(result[2], 43.0f);
    EXPECT_FLOAT_EQ(result[3], 50.0f);
}

// This case is just to check that the matrix multiplication works fully, this case will rarely occur in typical situations
TEST(test_MatMul, test_forward_2d_different_dimensions) {
    Tensor<float> input = tensor_mml<float>({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor<float> weight = tensor_mml<float>({4, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    MatMul<float> matmul(weight);

    Tensor<float> result = matmul.forward(input);
    
    // Test that the shape of result is 2x3
    EXPECT_EQ(result.get_shape().at(0), 2);
    EXPECT_EQ(result.get_shape().at(1), 3);

    EXPECT_FLOAT_EQ(result[0], 70.0f);
    EXPECT_FLOAT_EQ(result[1], 80.0f);
    EXPECT_FLOAT_EQ(result[2], 90.0f);
    EXPECT_FLOAT_EQ(result[3], 158.0f);
    EXPECT_FLOAT_EQ(result[4], 184.0f);
    EXPECT_FLOAT_EQ(result[5], 210.0f);
}


/* TEST(test_MatMul, test_forward_3d) {
    Tensor<float> input = tensor_mml<float>({2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> weight = tensor_mml<float>({2, 2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 5.0f, 6.0f, 7.0f, 8.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    MatMul<float> matmul(weight);

    Tensor<float> result = matmul.forward(input);
    
    // First slice
    EXPECT_FLOAT_EQ(result[0], 19.0f);
    EXPECT_FLOAT_EQ(result[1], 22.0f);
    EXPECT_FLOAT_EQ(result[2], 43.0f);
    EXPECT_FLOAT_EQ(result[3], 50.0f);
    
    // Second slice
    EXPECT_FLOAT_EQ(result[4], 19.0f);
    EXPECT_FLOAT_EQ(result[5], 22.0f);
    EXPECT_FLOAT_EQ(result[6], 43.0f);
    EXPECT_FLOAT_EQ(result[7], 50.0f);

    // Third slice
    EXPECT_FLOAT_EQ(result[8], 19.0f);
    EXPECT_FLOAT_EQ(result[9], 22.0f);
    EXPECT_FLOAT_EQ(result[10], 43.0f);
    EXPECT_FLOAT_EQ(result[11], 50.0f);
} */