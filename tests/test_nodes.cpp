#include <gtest/gtest.h>

#include <modularml>

TEST(test_node, test_ReLU_float) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f});
  auto original_X = tensor_mml_p<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;
  

  ReLUNode reluNode(x_string, y_string);
  reluNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";
  
  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";
  
  ASSERT_EQ(*result_ptr, *b);
  ASSERT_EQ(*input_ptr, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_ReLU_int32) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<int32_t>({3, 3}, {0, 5, 0, 10, 0, 15, 20, 0, 25});
  auto original_X = tensor_mml_p<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25});

  auto X = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25}));
  auto Y = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  //iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  ReLUNode reluNode(x_string, y_string);
  reluNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<int32_t>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";
  
  auto input_ptr = std::get<std::shared_ptr<Tensor<int32_t>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
  ASSERT_EQ(*input_ptr, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_TanH_float) {
  /**
   * @brief Expected Tensor after the TanH function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.7615941559557649f, 0.0f, 0.7615941559557649f, -0.9640275800758169f, 0.9640275800758169f, -0.9950547536867305f, 0.9950547536867305f, 0.999329299739067f, -0.999329299739067f});
  auto original_X = tensor_mml_p<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  //iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  TanHNode tanhNode(x_string, y_string);
  tanhNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";
  
  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
  ASSERT_EQ(*input_ptr, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_Swish_float) {
  /**
   * @brief Expected Tensor after the Swish function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});
  auto original_X = tensor_mml_p<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));
  
  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  //iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  SwishNode swishNode(x_string, y_string);
  swishNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";
  
  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_reshape_basic) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data tensor.
   */
  auto b = tensor_mml_p<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto data = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  auto shape = tensor_mml_p<int64_t>({2}, {2, 3});
  auto reshaped = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3}));

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
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(reshaped_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
}

TEST(test_node, test_reshape_high_dimensional) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data tensor.
   */
  auto b = tensor_mml_p<float>({2, 1, 3, 1}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

  auto data = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  auto shape = tensor_mml_p<int64_t>({4}, {2, 1, 3, 1});
  auto reshaped = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 1, 3, 1}));

  std::string data_string = "data";
  std::string shape_string = "shape";
  std::string reshaped_string = "reshaped";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  iomap[shape_string] = shape;
  //iomap[reshaped_string] = reshaped; Not mapping to test auto creation of output tensor

  reshapeNode reshapeNode(data_string, shape_string, reshaped_string);
  reshapeNode.forward(iomap);

  auto reshaped_it = iomap.find(reshaped_string);
  ASSERT_NE(reshaped_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(reshaped_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
}

TEST(test_node, test_Dropout_float) {
  /**
   * @brief If dropout is not in training mode, the output should be the same as the input.
   */
  auto data = tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});
  auto reference = data;
  auto output = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  std::string data_string = "data";
  std::string output_string = "output";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  iomap[output_string] = output;

  DropoutNode DropoutNode(data_string, output_string);
  DropoutNode.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *reference);
}

TEST(test_node, test_Dropout_random_float) {
  /**
   * @brief If dropout is not in training mode, the output should be the same as the input.
   */
  const array_mml<int> shape = generate_random_array_mml_integral<int>(3, 3, 3, 3);
  auto data = make_shared<Tensor_mml<float>>(generate_random_tensor<float>(shape, -5.0f, 5.0f));
  auto reference = data;
  auto output = make_shared<Tensor_mml<float>>(Tensor_mml<float>(shape));

  std::string data_string = "data";
  std::string output_string = "output";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  //iomap[output_string] = output; Not mapping to test auto creation of output tensor

  DropoutNode DropoutNode(data_string, output_string);
  DropoutNode.forward(iomap);

  auto output_it = iomap.find(output_string);
  ASSERT_NE(output_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";
  
  ASSERT_EQ(*result_ptr, *reference);
}

TEST(test_node, test_reshape_with_inferred_dimension) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data tensor.
   * This tests the automatic inference of one dimension using `-1`.
   */
  auto b = tensor_mml_p<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto data = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  auto shape = tensor_mml_p<int64_t>({2}, {-1, 3});
  auto reshaped = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3}));

  std::string data_string = "data";
  std::string shape_string = "shape";
  std::string reshaped_string = "reshaped";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  iomap[shape_string] = shape;
  //iomap[reshaped_string] = reshaped; Not mapping to test auto creation of output tensor

  reshapeNode reshapeNode(data_string, shape_string, reshaped_string);
  reshapeNode.forward(iomap);

  auto reshaped_it = iomap.find(reshaped_string);
  ASSERT_NE(reshaped_it, iomap.end()) << "Y tensor was not created";
  
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(reshaped_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
}
