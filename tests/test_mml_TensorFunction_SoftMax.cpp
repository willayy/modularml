#include <gtest/gtest.h>
#include <modularml>
#include <cmath>

/**
 * @brief Test SoftMax function with a 1D tensor.
 */
TEST(mml_TensorFunction_SoftMax, OneDimensionalTensor) {
    mml_TensorFunction_SoftMax softmax; // Default: last axis (-1)

    // Define input tensor
    Tensor<float> t1 = tensor_mml<float>({3}, {1.0f, 2.0f, 3.0f});
    
    // Compute expected SoftMax values
    float max_val = 3.0f;
    float exp1 = std::exp(1.0f - max_val);
    float exp2 = std::exp(2.0f - max_val);
    float exp3 = std::exp(3.0f - max_val);
    float sum = exp1 + exp2 + exp3;

    Tensor<float> expected = tensor_mml<float>({3}, {exp1 / sum, exp2 / sum, exp3 / sum});

    // Apply SoftMax
    Tensor<float> result = softmax.func(t1);

    // Validate output
    ASSERT_TRUE(tensors_are_close(result, expected));
}

/**
 * @brief Test SoftMax function with a 2D tensor.
 */
TEST(mml_TensorFunction_SoftMax, TwoDimensionalTensor) {
    mml_TensorFunction_SoftMax softmax(-1); // Apply along last axis

    // Define input tensor
    Tensor<float> t2 = tensor_mml<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    // Compute expected SoftMax values row-wise
    float max_val_row1 = 3.0f;
    float exp1 = std::exp(1.0f - max_val_row1);
    float exp2 = std::exp(2.0f - max_val_row1);
    float exp3 = std::exp(3.0f - max_val_row1);
    float sum1 = exp1 + exp2 + exp3;

    float max_val_row2 = 6.0f;
    float exp4 = std::exp(4.0f - max_val_row2);
    float exp5 = std::exp(5.0f - max_val_row2);
    float exp6 = std::exp(6.0f - max_val_row2);
    float sum2 = exp4 + exp5 + exp6;

    Tensor<float> expected = tensor_mml<float>({2, 3}, {
        exp1 / sum1, exp2 / sum1, exp3 / sum1,
        exp4 / sum2, exp5 / sum2, exp6 / sum2
    });

    // Apply SoftMax
    Tensor<float> result = softmax.func(t2);

    // Validate output
    ASSERT_TRUE(tensors_are_close(result, expected));
}