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

  // Check if the output tensor is correct
  ASSERT_EQ(*b, *C);
  // Check if the input tensors are unchanged
  ASSERT_EQ(*A, *A_ref);
  ASSERT_EQ(*B, *B_ref);
}

TEST(AddNodeTest, AddNodeTest_Broadcasting_float) {
  /**
   * @brief Expected result tensor.
   */
  auto b = tensor_mml_p<float>({1, 2}, {3.0f, 4.0f});  // A + B expected output after broadcasting

  /**
   * @brief Input tensors.
   */
  auto A = tensor_mml_p<float>({1, 2}, {1.0f, 2.0f});  // Input A (1x2)
  auto B = tensor_mml_p<float>({1, 1}, {2.0f});        // Input B (1x1) - needs broadcasting

  auto A_ref = A;
  auto B_ref = B;

  /**
   * @brief Output tensor.
   */
  auto C = tensor_mml_p<float>({1, 2});  // Output tensor (expected to match A's shape after broadcasting B)

  AddNode<float> addNode(A, B, C);
  addNode.forward();  // Should invoke broadcast addition

  // Ensure output tensor is correct
  ASSERT_EQ(*b, *C);
  // Ensure input tensors remain unchanged
  ASSERT_EQ(*A, *A_ref);
  ASSERT_EQ(*B, *B_ref);
}

TEST(AddNodeTest, AddNodeTest_Broadcasting_Complex_float) {
  /**
   * @brief Expected result tensor.
   * B should be broadcasted across dimensions [0] and [2] to match A.
   */
  auto b = tensor_mml_p<float>(
      {2, 3, 4},  // Shape (2x3x4)
      {
          3.0f, 4.0f, 5.0f, 6.0f, 9.0f, 10.0f, 11.0f, 12.0f, 15.0f, 16.0f, 17.0f, 18.0f,
          15.0f, 16.0f, 17.0f, 18.0f, 21.0f, 22.0f, 23.0f, 24.0f, 27.0f, 28.0f, 29.0f, 30.0f});

  /**
   * @brief Input tensors.
   * A is (2,3,4), B is (1,3,1) -> Needs broadcasting along dimensions [0] and [2]
   */
  auto A = tensor_mml_p<float>(
      {2, 3, 4},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f});

  auto B = tensor_mml_p<float>(
      {1, 3, 1},  // Shape (1x3x1) - needs broadcasting
      {
          2.0f, 4.0f, 6.0f  // Each row of A will be added with a different value
      });

  auto A_ref = A;  // Reference copy of A
  auto B_ref = B;  // Reference copy of B

  /**
   * @brief Output tensor.
   */
  auto C = tensor_mml_p<float>({2, 3, 4});  // Output tensor

  AddNode<float> addNode(A, B, C);
  addNode.forward();  // Should invoke broadcast addition

  // Ensure output tensor is correct
  ASSERT_EQ(*b, *C);

  // Ensure input tensors remain unchanged
  ASSERT_EQ(*A, *A_ref);
  ASSERT_EQ(*B, *B_ref);
}

TEST(AddNodeTest, AddNodeTest_Broadcasting_Simple_A_Broadcasts) {
  /**
   * @brief Expected result tensor.
   * A should be broadcasted along dimension 0 to match B.
   */
  auto b = tensor_mml_p<float>(
      {2, 3},  // Shape (2x3)
      {
          3.0f, 5.0f, 7.0f,  // A's values (1,2,3) broadcasted across row 0
          6.0f, 8.0f, 10.0f  // A's values (1,2,3) broadcasted across row 1
      });

  /**
   * @brief Input tensors.
   * A is (1,3), B is (2,3) -> A needs broadcasting along dimension 0
   */
  auto A = tensor_mml_p<float>(
      {1, 3},  // Shape (1x3)
      {
          1.0f, 2.0f, 3.0f  // Will be expanded to match B's shape (2,3)
      });

  auto B = tensor_mml_p<float>(
      {2, 3},  // Shape (2x3)
      {
          2.0f, 3.0f, 4.0f,
          5.0f, 6.0f, 7.0f});

  auto A_ref = A;  // Reference copy of A
  auto B_ref = B;  // Reference copy of B

  /**
   * @brief Output tensor.
   */
  auto C = tensor_mml_p<float>({2, 3});  // Output tensor

  AddNode<float> addNode(A, B, C);
  addNode.forward();  // Should invoke broadcast addition

  // Ensure output tensor is correct
  ASSERT_EQ(*b, *C);

  // Ensure input tensors remain unchanged
  ASSERT_EQ(*A, *A_ref);
  ASSERT_EQ(*B, *B_ref);
}

TEST(AddNodeTest, AddNodeTest_Broadcasting_IncompatibleShapes) {
  /**
   * @brief Input tensors with incompatible shapes.
   * A is (2,3), B is (3,2) -> Cannot be broadcasted.
   */
  auto A = tensor_mml_p<float>(
      {2, 3},
      {1.0f, 2.0f, 3.0f,
       4.0f, 5.0f, 6.0f});

  auto B = tensor_mml_p<float>(
      {3, 2},
      {7.0f, 8.0f,
       9.0f, 10.0f,
       11.0f, 12.0f});

  /**
   * @brief Output tensor (should never be created due to the error).
   */
  auto C = tensor_mml_p<float>({2, 3});  // This should not be used.

  AddNode<float> addNode(A, B, C);

  /**
   * @brief Expect a runtime error due to incompatible broadcasting.
   */
  EXPECT_THROW(addNode.forward(), std::runtime_error);
}
