#include <gtest/gtest.h>

#include <modularml>

TEST(test_mul, test_constructor) {
    Tensor<float> weights = tensor_mml<float>({2, 2}, {0.1, 0.2, 0.3, 0.4});
    MatMul<float> matmul = MatMul<float>(weights);
    
    EXPECT_EQ(matmul.get_weights().get_shape(), vector<int>({2, 2}));
}