#include <gtest/gtest.h>

#include <modularml>

const MaxPoolingLayer<float> max_pool(std::vector<int>{2, 2}, std::vector<int>{2, 2});
const AvgPoolingLayer<float> avg_pool(std::vector<int>{2, 2}, std::vector<int>{2, 2});

TEST(test_mml_pooling, test_max_pooling_1) {
  Tensor<float> t = tensor_mml<float>({1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  float value = t[{0, 0, 0, 0}];
  ASSERT_EQ(value, 1);
}

TEST(test_mml_pooling, test_max_pooling_2) {
  const Tensor<float> input = tensor_mml<float>({1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const Tensor<float> output = tensor_mml<float>({1, 2, 2, 1}, {6, 8, 14, 16});

  Tensor<float> result = max_pool.forward(input);

  ASSERT_EQ(result, output);
}