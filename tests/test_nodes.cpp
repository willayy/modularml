#include <gtest/gtest.h>

#include <modularml>

TEST(test_node, test_ReLU_float) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f});
  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  ReLUNode<float> reluNode(X, Y);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
}

TEST(test_node, test_ReLU_int32) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<int32_t>({3, 3}, {0, 5, 0, 10, 0, 15, 20, 0, 25});

  auto X = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25}));
  auto Y = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}));

  ReLUNode<int32_t> reluNode(X, Y);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
}

TEST(test_node, test_TanH_float) {
  /**
   * @brief Expected Tensor after the TanH function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.7615941559557649f, 0.0f, 0.7615941559557649f, -0.9640275800758169f, 0.9640275800758169f, -0.9950547536867305f, 0.9950547536867305f, 0.999329299739067f, -0.999329299739067f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  TanHNode<float> tanhNode(X, Y);
  tanhNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
}

TEST(test_node, test_Swish_float) {
  /**
   * @brief Expected Tensor after the Swish function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  SwishNode<float> swishNode(X, Y);
  swishNode.forward();

  ASSERT_TRUE(tensors_are_close(*Y, *b));
}

TEST(test_node, test_reshape_float) {
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

  // Test where the setInputs method is used with a high-dimensional reshape
  data = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));

  // Complex reshape: Change (3,2) → (2,1,3,1) for multi-dimensional testing
  shape = tensor_mml_p<int64_t>({4}, {2, 1, 3, 1});

  /**
   * @brief Expected Tensor after the Reshape function is applied to the data tensor.
   */
  b = tensor_mml_p<float>({2, 1, 3, 1}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

  array_mml<GeneralDataTypes> inputs({data, shape});
  reshapeNode.setInputs(inputs);

  reshapeNode.forward();

  // Check if the reshaped tensor matches expected output
  ASSERT_EQ(*reshaped, *b);

  /**
   * @brief Expected Tensor after the Reshape function is applied to the data tensor.
   * This tests the automatic inference of one dimension using `-1`.
   */
  // Expected output tensor after reshape (same values, different shape)
  b = tensor_mml_p<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // Input tensor (3,2)
  data = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

  // Shape tensor with `-1`, meaning the missing dimension should be inferred
  shape = tensor_mml_p<int64_t>({2}, {-1, 3});  // ONNX rule: `-1` is automatically computed

  // Reshaped output tensor (expected to become {2,3})

  inputs = array_mml<GeneralDataTypes>({data, shape});
  reshapeNode.setInputs(inputs);

  reshapeNode.forward();

  // Ensure that the reshaped tensor matches the expected output
  ASSERT_EQ(*reshaped, *b);
}