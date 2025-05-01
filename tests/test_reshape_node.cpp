#include <gtest/gtest.h>

#include <modularml>

TEST(test_node, test_reshape_basic) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data
   * tensor.
   */
  auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{2UL, 3UL}, array_mml<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto data = std::make_shared<Tensor<float>>(array_mml<size_t>{3UL, 2UL}, array_mml<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto shape = std::make_shared<Tensor<int64_t>>(array_mml<size_t>{2}, array_mml<int64_t>{2, 3});
  auto reshaped = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3});

  std::string data_string = "data";
  std::string shape_string = "shape";
  std::string reshaped_string = "reshaped";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  iomap[shape_string] = shape;
  iomap[reshaped_string] = reshaped;

  reshapeNode reshapeNode(data_string, shape_string, reshaped_string);
  reshapeNode.forward(iomap);

  auto reshaped_it = iomap.find(reshaped_string);
  ASSERT_NE(reshaped_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr =
      std::get<std::shared_ptr<Tensor<float>>>(reshaped_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
}

TEST(test_node, test_reshape_high_dimensional) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data
   * tensor.
   */
  auto b = std::make_shared<Tensor<float>>(
    array_mml<size_t>{2, 1, 3, 1}, 
    array_mml<float>{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

  auto data = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 2}, 
    array_mml<float>{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  auto shape = std::make_shared<Tensor<int64_t>>(array_mml<size_t>{4}, array_mml<int64_t>{2, 1, 3, 1});
  auto reshaped = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 1, 3, 1});

  std::string data_string = "data";
  std::string shape_string = "shape";
  std::string reshaped_string = "reshaped";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  iomap[shape_string] = shape;
  // iomap[reshaped_string] = reshaped; Not mapping to test auto creation of
  // output tensor

  reshapeNode reshapeNode(data_string, shape_string, reshaped_string);
  reshapeNode.forward(iomap);

  auto reshaped_it = iomap.find(reshaped_string);
  ASSERT_NE(reshaped_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr =
      std::get<std::shared_ptr<Tensor<float>>>(reshaped_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
}

TEST(test_node, test_reshape_with_inferred_dimension) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data
   * tensor. This tests the automatic inference of one dimension using `-1`.
   */
  auto b = std::make_shared<Tensor<float>>(
    array_mml<size_t>{2, 3}, 
    array_mml<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto data = std::make_shared<Tensor<float>>(
    array_mml<size_t>{3, 2}, 
    array_mml<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto shape = std::make_shared<Tensor<int64_t>>(array_mml<size_t>{2}, array_mml<int64_t>{-1, 3});
  auto reshaped = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3});

  std::string data_string = "data";
  std::string shape_string = "shape";
  std::string reshaped_string = "reshaped";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  iomap[shape_string] = shape;
  // iomap[reshaped_string] = reshaped; Not mapping to test auto creation of
  // output tensor

  reshapeNode reshapeNode(data_string, shape_string, reshaped_string);
  reshapeNode.forward(iomap);

  auto reshaped_it = iomap.find(reshaped_string);
  ASSERT_NE(reshaped_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr =
      std::get<std::shared_ptr<Tensor<float>>>(reshaped_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
}
