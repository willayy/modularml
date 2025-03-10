#include <gtest/gtest.h>

#include <modularml>

TEST(test_mml_pooling, test_max_pooling_without_padding_1) {

  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 2, 1});
  const shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 2, 2, 1}, {6, 8, 14, 16});

  MaxPoolingNode_mml<float> max_pool_without_padding = MaxPoolingNode_mml(
      std::vector<int>{2, 2}, std::vector<int>{2, 2}, input, output);

  max_pool_without_padding.forward();
  auto result = max_pool_without_padding.getOutputs();
  auto tensor_ptr = std::get<std::shared_ptr<Tensor<float>>>(result[0]);

  ASSERT_EQ((*exp_output), (*tensor_ptr));
}

TEST(test_mml_pooling, test_max_pooling_with_padding_1) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 2, 1});
  const shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 2, 2, 1}, {6, 8, 14, 16});

  MaxPoolingNode_mml<float> max_pool_with_padding = MaxPoolingNode_mml(
      std::vector<int>{2, 2}, std::vector<int>{2, 2}, input, output, "same");

  max_pool_with_padding.forward();
  auto result = max_pool_with_padding.getOutputs();
  auto tensor_ptr = std::get<std::shared_ptr<Tensor<float>>>(result[0]);

  ASSERT_EQ((*exp_output), (*tensor_ptr));
}

TEST(test_mml_pooling, test_avg_pooling_with_padding_1) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 2, 1});
  const shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 2, 2, 1}, {3.5, 5.5, 11.5, 13.5});

  AvgPoolingNode_mml<float> avg_pool_without_padding = AvgPoolingNode_mml(
      std::vector<int>{2, 2}, std::vector<int>{2, 2}, input, output);

  avg_pool_without_padding.forward();
  auto result = avg_pool_without_padding.getOutputs();
  auto tensor_ptr = std::get<std::shared_ptr<Tensor<float>>>(result[0]);

  ASSERT_EQ((*exp_output), (*tensor_ptr));
}

TEST(test_mml_pooling, test_max_pooling_with_padding_2) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 4, 5, 1},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 3, 1});
  const shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 2, 3, 1}, {7, 9, 10, 17, 19, 20});

  MaxPoolingNode_mml<float> max_pool_with_padding = MaxPoolingNode_mml(
      std::vector<int>{2, 2}, std::vector<int>{2, 2}, input, output, "same");
  max_pool_with_padding.forward();
  auto result = max_pool_with_padding.getOutputs();
  auto tensor_ptr = std::get<std::shared_ptr<Tensor<float>>>(result[0]);

  ASSERT_EQ((*exp_output), (*tensor_ptr));
}

TEST(test_mml_pooling, test_avg_pooling_with_padding_2) {
  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 4, 5, 1},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 2, 3, 1});
  const shared_ptr<Tensor<float>> exp_output =
      tensor_mml_p<float>({1, 2, 3, 1}, {4.0, 6.0, 3.75, 14.0, 16.0, 8.75});

  AvgPoolingNode_mml<float> avg_pool_with_padding = AvgPoolingNode_mml(
      std::vector<int>{2, 2}, std::vector<int>{2, 2}, input, output, "same");
  avg_pool_with_padding.forward();
  auto result = avg_pool_with_padding.getOutputs();
  auto tensor_ptr = std::get<std::shared_ptr<Tensor<float>>>(result[0]);

  ASSERT_EQ((*exp_output), (*tensor_ptr));
}

TEST(test_mml_pooling, test_max_pooling_with_padding_3) {

  const shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 4, 5, 1},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  const shared_ptr<Tensor<float>> output = tensor_mml_p<float>({1, 4, 3, 1});
  const shared_ptr<Tensor<float>> exp_output = tensor_mml_p<float>(
      {1, 4, 3, 1}, {7, 9, 10, 12, 14, 15, 17, 19, 20, 17, 19, 20});

  MaxPoolingNode_mml<float> max_pool_padding_stride = MaxPoolingNode_mml(
      std::vector<int>{2, 3}, std::vector<int>{1, 2}, input, output, "same");
  max_pool_padding_stride.forward();
  auto result = max_pool_padding_stride.getOutputs();
  auto tensor_ptr = std::get<std::shared_ptr<Tensor<float>>>(result[0]);

  ASSERT_EQ((*exp_output), (*tensor_ptr));
}