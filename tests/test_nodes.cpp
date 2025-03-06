#include <gtest/gtest.h>

#include <modularml>

TEST(test_node, test_ReLU_float) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f});

  auto X = make_shared<DataTypes>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<DataTypes>(Tensor_mml<float>({3, 3}));

  ReLUNode reluNode(X, Y);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  auto& result = get<Tensor_mml<float>>(*Y);
  ASSERT_EQ(result, *b);
}

TEST(test_node, test_ReLU_int32) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<int32_t>({3, 3}, {0, 5, 0, 10, 0, 15, 20, 0, 25});

  auto X = make_shared<DataTypes>(Tensor_mml<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25}));
  auto Y = make_shared<DataTypes>(Tensor_mml<int32_t>({3, 3}));

  ReLUNode reluNode(X, Y);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  auto& result = get<Tensor_mml<int32_t>>(*Y);
  ASSERT_EQ(result, *b);
}
