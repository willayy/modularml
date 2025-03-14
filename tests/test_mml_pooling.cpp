#include <gtest/gtest.h>

#include <modularml>

TEST(test_mml_pooling, test_max_pool_1) {

  shared_ptr<Tensor<float>> input = tensor_mml_p<float>(
      {1, 1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  shared_ptr<Tensor<float>> output =
      tensor_mml_p<float>({1, 1, 2, 2}, {6, 8, 14, 16});
  shared_ptr<Tensor<float>> output_indices =
      tensor_mml_p<float>({1, 1, 2, 2}, {5, 7, 13, 15});

  MaxPoolingNode_mml<float> max_pool = MaxPoolingNode_mml<float>(
      {2, 2}, {2, 2}, input, "NOTSET", 0, {1, 1}, {0, 0}, 0);

  max_pool.forward();

  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&max_pool.output[0])) {
    shared_ptr<Tensor<float>> real_output = *tensor_ptr;
    std::cerr << "Resulting tensor: " << real_output->to_string() << std::endl;
    ASSERT_EQ(*real_output, *output);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl;
  }
  if (auto tensor_ptr =
          std::get_if<std::shared_ptr<Tensor<float>>>(&max_pool.output[1])) {
    shared_ptr<Tensor<float>> real_output_indices = *tensor_ptr;
    std::cerr << "Resulting tensor: " << real_output_indices->to_string()
              << std::endl;
    ASSERT_EQ(*real_output_indices, *output_indices);
  } else {
    std::cerr << "Error: The variant does not hold the expected type"
              << std::endl;
  }
}
