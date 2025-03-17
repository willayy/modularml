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

  ReLUNode<float> reluNode(X, Y);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
  ASSERT_EQ(*X, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_ReLU_int32) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<int32_t>({3, 3}, {0, 5, 0, 10, 0, 15, 20, 0, 25});
  auto original_X = tensor_mml_p<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25});

  auto X = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}, {-9, 5, -3, 10, -2, 15, 20, -6, 25}));
  auto Y = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}));

  ReLUNode<int32_t> reluNode(X, Y);

  // Testing the use of setInput method here as well:
  X = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25}));
  array_mml<GeneralDataTypes> inputs{X};
  reluNode.setInputs(inputs);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
  ASSERT_EQ(*X, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_TanH_float) {
  /**
   * @brief Expected Tensor after the TanH function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.7615941559557649f, 0.0f, 0.7615941559557649f, -0.9640275800758169f, 0.9640275800758169f, -0.9950547536867305f, 0.9950547536867305f, 0.999329299739067f, -0.999329299739067f});
  auto original_X = tensor_mml_p<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  TanHNode<float> tanhNode(X, Y);
  tanhNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
  ASSERT_EQ(*X, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_Swish_float) {
  /**
   * @brief Expected Tensor after the Swish function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});
  auto original_X = tensor_mml_p<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  SwishNode<float> swishNode(X, Y);
  swishNode.forward();

  ASSERT_TRUE(tensors_are_close(*Y, *b));
  ASSERT_EQ(*X, *original_X);  // Ensure the input tensor is intact
}

TEST(test_node, test_reshape_basic) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data tensor.
   */
  auto b = tensor_mml_p<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  auto data = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  auto shape = tensor_mml_p<int64_t>({2}, {2, 3});
  auto reshaped = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3}));

  reshapeNode<float> reshapeNode(data, shape, reshaped);
  reshapeNode.forward();

  ASSERT_EQ(*reshaped, *b);
}

TEST(test_node, test_reshape_high_dimensional) {
  /**
   * @brief Expected Tensor after the Reshape function is applied to the data tensor.
   */
  auto b = tensor_mml_p<float>({2, 1, 3, 1}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

  auto data = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  auto shape = tensor_mml_p<int64_t>({4}, {2, 1, 3, 1});
  auto reshaped = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 1, 3, 1}));

  reshapeNode<float> reshapeNode(data, shape, reshaped);
  array_mml<GeneralDataTypes> inputs({data, shape}); // This is because we also want to test the setInputs function
  reshapeNode.setInputs(inputs);
  reshapeNode.forward();

  ASSERT_EQ(*reshaped, *b);
}

TEST(test_node, test_Dropout_float) {
  /**
   * @brief If dropout is not in training mode, the output should be the same as the input.
   */
  auto data = tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});
  auto refrence = data;
  auto output = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  DropoutNode<float> DropoutNode(data,output);
  DropoutNode.forward();

  ASSERT_EQ(*output, *refrence);
}

TEST(test_node, test_Dropout_random_float) {
  /**
   * @brief If dropout is not in training mode, the output should be the same as the input.
   */
  const array_mml<int> shape = generate_random_array_mml_integral<int>(3, 3, 3, 3);
  auto data = make_shared<Tensor_mml<float>>(generate_random_tensor<float>(shape, -5.0f, 5.0f));
  auto reference = data;
  auto output = make_shared<Tensor_mml<float>>(Tensor_mml<float>(shape));

  DropoutNode<float> dropoutNode(data, output);
  dropoutNode.forward();

  ASSERT_EQ(*output, *reference);
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

  reshapeNode<float> reshapeNode(data, shape, reshaped);
  array_mml<GeneralDataTypes> inputs({data, shape}); // This is because we also want to test the setInputs function
  reshapeNode.setInputs(inputs);
  reshapeNode.forward();

  ASSERT_EQ(*reshaped, *b);
}

TEST(test_node, test_Softmax_float) {
  /**
   * @brief Expected Tensor after the Softmax function is applied.
   */
  auto original_X = tensor_mml_p<float>({2, 3}, {1.0f, 2.0f, 3.0f, 2.0f, 3.0f, 4.0f});
  auto expected_Y = tensor_mml_p<float>({2, 3}, {0.09003057, 0.24472847, 0.66524096,
                                                0.09003057, 0.24472847, 0.66524096});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3}, {1.0f, 2.0f, 3.0f, 2.0f, 3.0f, 4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3}));

  SoftmaxNode<float> softmaxNode(X, Y, -1);
  softmaxNode.forward();

  // Retrieve output via getOutput()
  auto result = softmaxNode.getOutput();

  // Check values
  for (int i = 0; i < result->get_size(); i++) {
      EXPECT_NEAR((*result)[i], (*expected_Y)[i], 1e-5);
  }
}

TEST(test_node, test_Softmax_int32) {
  /**
   * @brief Softmax should work for int32_t but behave similarly to float casting.
   */
  auto original_X = tensor_mml_p<int32_t>({2, 3}, {1, 2, 3, 2, 3, 4});
  auto expected_Y = tensor_mml_p<int32_t>({2, 3}, {0, 0, 1, 0, 0, 1});

  auto X = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({2, 3}, {1, 2, 3, 2, 3, 4}));
  auto Y = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({2, 3}));

  SoftmaxNode<int32_t> softmaxNode(X, Y, -1);
  softmaxNode.forward();

  // Retrieve output via getOutput()
  auto result = softmaxNode.getOutput();

  // Check values (integer-based softmax rounds values, hence checking 0 or 1)
  for (int i = 0; i < result->get_size(); i++) {
      EXPECT_LE((*result)[i], 1);
      EXPECT_GE((*result)[i], 0);
  }
}