#include <gtest/gtest.h>

#include <modularml>

TEST(test_node, test_ReLU_float) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<float>(
      {3, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f});
  auto original_X = tensor_mml_p<float>(
      {3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>(
      {3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
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
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_ReLU_int32) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<int32_t>({3, 3}, {0, 5, 0, 10, 0, 15, 20, 0, 25});
  auto original_X =
      tensor_mml_p<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25});

  auto X = make_shared<Tensor_mml<int32_t>>(
      Tensor_mml<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25}));
  auto Y = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  // iomap[y_string] = Y; Not mapping to test auto creation of output tensor

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
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_ReLU_double) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b =
      tensor_mml_p<double>({3, 3}, {0.0f, 0.0000000000000001f,
                                    std::numeric_limits<double>::infinity(),
                                    0.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f});
  auto original_X = tensor_mml_p<double>(
      {3, 3}, {-999999999999999999999999999.0f, 0.0000000000000001f,
               std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity(), 2.0f, -3.0f, 3.0f,
               4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<double>>(Tensor_mml<double>(
      {3, 3}, {-999999999999999999999999999.0f, 0.0000000000000001f,
               std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity(), 2.0f, -3.0f, 3.0f,
               4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<double>>(Tensor_mml<double>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  ReLUNode reluNode(x_string, y_string);
  reluNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<double>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<double>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_EQ(*result_ptr, *b);
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_TanH_float) {
  /**
   * @brief Expected Tensor after the TanH function is applied to each element.
   */
  auto b = tensor_mml_p<float>(
      {3, 3}, {-0.7615941559557649f, 0.0f, 0.7615941559557649f,
               -0.9640275800758169f, 0.9640275800758169f, -0.9950547536867305f,
               0.9950547536867305f, 0.999329299739067f, -0.999329299739067f});
  auto original_X = tensor_mml_p<float>(
      {3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>(
      {3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  // iomap[y_string] = Y; Not mapping to test auto creation of output tensor

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
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_TanH_double) {
  /**
   * @brief Expected Tensor after the TanH function is applied to each element.
   */
  auto b = tensor_mml_p<double>(
      {3, 3}, {-1.0f, 0.0f, 0.7615941559557649f, -0.9640275800758169f,
               0.9640275800758169f, -0.9950547536867305f, 0.9950547536867305f,
               0.999329299739067f, 1.0f});
  auto original_X =
      tensor_mml_p<double>({3, 3}, {-std::numeric_limits<double>::infinity(),
                                    0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f,
                                    std::numeric_limits<double>::infinity()});

  auto X = make_shared<Tensor_mml<double>>(
      Tensor_mml<double>({3, 3}, {-std::numeric_limits<double>::infinity(),
                                  0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f,
                                  std::numeric_limits<double>::infinity()}));
  auto Y = make_shared<Tensor_mml<double>>(Tensor_mml<double>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  // iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  TanHNode tanhNode(x_string, y_string);
  tanhNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<double>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<double>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_Swish_float) {
  /**
   * @brief Expected Tensor after the Swish function is applied to each element.
   */
  auto b =
      tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f,
                                   -0.1423f, 2.8577f, 3.9281f, -0.0719f});
  auto original_X = tensor_mml_p<float>(
      {3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>(
      {3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  // iomap[y_string] = Y; Not mapping to test auto creation of output tensor

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
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_reshape_basic) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data
   * tensor.
   */
  auto b =
      tensor_mml_p<float>({2UL, 3UL}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto data = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3UL, 2UL}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
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
  auto b = tensor_mml_p<float>({2, 1, 3, 1},
                               {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

  auto data = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  auto shape = tensor_mml_p<int64_t>({4}, {2, 1, 3, 1});
  auto reshaped =
      make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 1, 3, 1}));

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

TEST(test_node, test_Dropout_float) {
  /**
   * @brief If dropout is not in training mode, the output should be the same as
   * the input.
   */
  auto data =
      tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f,
                                   -0.1423f, 2.8577f, 3.9281f, -0.0719f});
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
   * @brief If dropout is not in training mode, the output should be the same as
   * the input.
   */
  const array_mml<uli> shape =
      generate_random_array_mml_integral<uli>(3, 3, 3, 3);
  auto data = make_shared<Tensor_mml<float>>(
      generate_random_tensor<float>(shape, -5.0f, 5.0f));
  auto reference = data;
  auto output = make_shared<Tensor_mml<float>>(shape);

  std::string data_string = "data";
  std::string output_string = "output";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[data_string] = data;
  // iomap[output_string] = output; Not mapping to test auto creation of output
  // tensor

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
   * @brief Expected Tensor after the Reshape function is applied to the data
   * tensor. This tests the automatic inference of one dimension using `-1`.
   */
  auto b = tensor_mml_p<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto data = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  auto shape = tensor_mml_p<int64_t>({2}, {-1, 3});
  auto reshaped = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3}));

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

TEST(test_node, test_Sigmoid_float) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 2}, {0.731059f, 0.880797f, 0.952574f,
                                        0.982014f, 0.993307f, 0.997527f});
  auto original_X =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto X = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  // iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  SigmoidNode sigmoidNode(x_string, y_string);
  sigmoidNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_Gelu_float) {
  /**
   * @brief Expected Tensor after the Gelu function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 2}, {0.841344f, 1.954499f, 2.995950f,
                                        -0.158655f, -0.0455003f, -0.004049f});
  auto original_X =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});

  auto X = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  // iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  GeluNode geluNode(x_string, y_string, "none");
  geluNode.forward(iomap);

  // Retrieve the tensor from the shared pointer Y

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact

  b = tensor_mml_p<float>({3, 2}, {0.841192f, 1.9546f, 2.99636f, -0.158808f,
                                   -0.045402f, -0.003637f});
  geluNode = GeluNode(x_string, y_string, "tanh");
  geluNode.forward(iomap);

  y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
}

TEST(test_node, test_Gelu_random_float) {
  /**
   * @brief Expected Tensor after the Gelu function is applied to each element.
   */
  auto b = tensor_mml_p<float>(
      {3, 2}, {-0.001011f, -0.169019f, 7.991757f, -0.036949f, 0.0f, 0.0f});
  auto original_X =
      tensor_mml_p<float>({3, 2}, {-3.436826f, -0.819629f, 7.991757f,
                                   -2.107639f, -7.18764f, -6.513006f});

  auto X = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {-3.436826f, -0.819629f, 7.991757f, -2.107639f,
                                 -7.18764f, -6.513006f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  // iomap[y_string] = Y; Not mapping to test auto creation of output tensor

  GeluNode geluNode(x_string, y_string, "none");
  geluNode.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_leaky_relu_float) {
  /**
   * @brief Expected Tensor after the LeakyReLU function is applied to each
   * element.
   */
  auto b =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.0f, 3.0f, -0.02f, -0.04f, -0.06f});

  auto original_X =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});

  auto X = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  LeakyReLUNode leakyReLU(x_string, y_string, 0.02);
  leakyReLU.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact //
                                      // Ensure the input tensor is intact
}

TEST(test_node, test_leaky_relu_random_float) {
  /**
   * @brief Expected Tensor after the LeakyReLU function is applied to each
   * element.
   */

  auto b = tensor_mml_p<float>({3, 2}, {1.491582f, 3.279023f, 8.310189f,
                                        -0.224878f, -0.0481f, 7.324412f});
  auto original_X =
      tensor_mml_p<float>({3, 2}, {1.491582f, 3.279023f, 8.310189f, -7.495929f,
                                   -1.602100f, 7.324412f});

  auto X = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {1.491582f, 3.279023f, 8.310189f, -7.495929f,
                                 -1.602100f, 7.324412f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  LeakyReLUNode leakyReLU(x_string, y_string, 0.03);
  leakyReLU.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_ELUNode_float) {

  /**
   * @brief Expected Tensor after the ELU function is applied to each element.
   */
  auto b = tensor_mml_p<float>(
      {3, 2}, {1.0f, 2.0f, 3.0f, -1.264241f, -1.729329f, -1.900425f});

  auto original_X =
      tensor_mml_p<float>({3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});

  auto X = make_shared<Tensor_mml<float>>(
      Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  ELUNode elu_node(x_string, y_string, 2.0f);
  elu_node.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  auto x_it = iomap.find(x_string);
  ASSERT_NE(x_it, iomap.end()) << "Y tensor was not created";

  auto input_ptr = std::get<std::shared_ptr<Tensor<float>>>(x_it->second);
  ASSERT_NE(input_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
  ASSERT_EQ(*input_ptr, *original_X); // Ensure the input tensor is intact
}

TEST(test_node, test_ELUNode_random_float) {

  /**
   * @brief Expected Tensor after the ELU function is applied to each element.
   */

  auto b = tensor_mml_p<float>({3, 2}, {-0.197959f, -0.199985f, -0.191696f,
                                        -0.172574f, -0.199538f, 7.627019});
  auto X = tensor_mml_p<float>({3, 2}, {-4.584662f, -9.531804f, -3.181585f,
                                        -1.986814f, -6.069519f, 7.627019f});

  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}));

  std::string x_string = "X";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[x_string] = X;
  iomap[y_string] = Y;

  ELUNode elu_node(x_string, y_string, 0.2f);
  elu_node.forward(iomap);

  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end()) << "Y tensor was not created";

  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);
  ASSERT_NE(result_ptr, nullptr) << "Failed to get Y tensor";

  ASSERT_TRUE(tensors_are_close(*result_ptr, *b));
}
