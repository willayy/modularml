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