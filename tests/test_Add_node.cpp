#include <gtest/gtest.h>

#include <modularml>

TEST(AddNodeTest, AddNodeTest_float) {
  /**
   * @brief Expected result tensor.
   */
  auto b = tensor_mml_p<float>({1, 2}, {3.0f, 5.0f});  // A + B expected output

  /**
   * @brief input tensors.
   */
  auto A = tensor_mml_p<float>({1, 2}, {1.0f, 2.0f});  // Input A
  auto B = tensor_mml_p<float>({1, 2}, {2.0f, 3.0f});  // Input B

  auto A_ref = A;
  auto B_ref = B;

  /**
   * @brief output tensor.
   */
  auto C = tensor_mml_p<float>({1, 2});

  AddNode<float> addNode(A, B, C);
  addNode.forward();

  ASSERT_EQ(*b, *C);
  // Check if the input tensors are unchanged
  ASSERT_EQ(*A, *A_ref);
  ASSERT_EQ(*B, *B_ref);
}