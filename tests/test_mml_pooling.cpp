#include <gtest/gtest.h>

#include <modularml>

const MaxPoolingLayer<float> max_pool_without_padding(std::vector<int>{2, 2}, std::vector<int>{2, 2});
const AvgPoolingLayer<float> avg_pool_without_padding(std::vector<int>{2, 2}, std::vector<int>{2, 2});

const MaxPoolingLayer<float> max_pool_with_padding(std::vector<int>{2, 2}, std::vector<int>{2, 2}, "same");
const AvgPoolingLayer<float> avg_pool_with_padding(std::vector<int>{2, 2}, std::vector<int>{2, 2}, "same");

TEST(test_mml_pooling, test_max_pooling_without_padding_1) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>({1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 2, 1}, {6, 8, 14, 16});

  shared_ptr<Tensor<float>> result = max_pool_without_padding.forward(input);

  ASSERT_EQ((*result), (*output));
}

TEST(test_mml_pooling, test_max_pooling_with_padding_1) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>({1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 2, 1}, {6, 8, 14, 16});

  shared_ptr<Tensor<float>> result = max_pool_with_padding.forward(input);

  ASSERT_EQ((*result), (*output));
}

TEST(test_mml_pooling, test_avg_pooling_with_padding_1) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>({1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 2, 1}, {3.5, 5.5, 11.5, 13.5});

  shared_ptr<Tensor<float>> result = avg_pool_without_padding.forward(input);

  ASSERT_EQ((*result), (*output));
}

TEST(test_mml_pooling, test_avg_pooling_without_padding_1) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>({1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 2, 1}, {3.5, 5.5, 11.5, 13.5});

  shared_ptr<Tensor<float>> result = avg_pool_with_padding.forward(input);

  ASSERT_EQ((*result), (*output));
}

TEST(test_mml_pooling, test_max_pooling_with_padding_2) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>({1, 4, 5, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 3, 1}, {7, 9, 10, 17, 19, 20});

  shared_ptr<Tensor<float>> result = max_pool_with_padding.forward(input);

  ASSERT_EQ((*result), (*output));
}

TEST(test_mml_pooling, test_avg_pooling_with_padding_2) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>({1, 4, 5, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 3, 1}, {4.0, 6.0, 3.75, 14.0, 16.0, 8.75});

  shared_ptr<Tensor<float>> result = avg_pool_with_padding.forward(input);

  ASSERT_EQ((*result), (*output));
}

TEST(test_mml_pooling, test_max_pooling_with_padding_3) {
  const MaxPoolingLayer<float> max_pool_padding_stride(std::vector<int>{2, 3}, std::vector<int>{1, 2}, "same");

  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>({1, 4, 5, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 4, 3, 1}, {7, 9, 10, 12, 14, 15, 17, 19, 20, 17, 19, 20});

  shared_ptr<Tensor<float>> result = max_pool_padding_stride.forward(input);

  ASSERT_EQ((*result), (*output));
}