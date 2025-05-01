#include <gtest/gtest.h>

#include "nodes/matmul.hpp"

TEST(MatMulNode_test, test_forward) {
  // Define dimensions: M = 2, K = 3, N = 2.
  array_mml<size_t> shapeA({2, 3});
  array_mml<size_t> shapeB({3, 2});
  array_mml<size_t> shapeY({2, 2});  // Output shape is [M, N]

  // Wrap each tensor in a shared pointer.
  auto A_ptr = std::make_shared<Tensor<float>>(shapeA, array_mml<float>{1, 2, 3, 4, 5, 6});
  auto B_ptr = std::make_shared<Tensor<float>>(shapeB, array_mml<float>{7, 8, 9, 10, 11, 12});
  auto Y_ptr = std::make_shared<Tensor<float>>(shapeY);
  Y_ptr->fill(0.0f);  // Initialize Y to zero

  // Setup the iomap with tensor names
  std::string a_string = "A";
  std::string b_string = "B";
  std::string y_string = "Y";
  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[a_string] = A_ptr;
  iomap[b_string] = B_ptr;
  iomap[y_string] =
      Y_ptr;  // Not mapping to test auto creation of output tensor

  // Construct the GemmNode with alpha=1.0, beta=0.0, no transposition
  MatMulNode node(a_string, b_string, y_string);

  // Run the forward pass
  node.forward(iomap);

  // Retrieve the result from iomap
  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end())
      << "Output tensor Y not found in iomap after forward pass";

  // Extract the shared pointer to the output tensor
  auto result_ptr = std::get<std::shared_ptr<Tensor<float>>>(y_it->second);

  // Expected result:
  // First row: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58,
  //            1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64.
  // Second row: 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139,
  //             4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154.
  auto expected = std::make_shared<Tensor<float>>(shapeY, array_mml<float>{58.0f, 64.0f, 139.0f, 154.0f});

  // Compare each element of the result with the expected value.
  for (int i = 0; i < expected->get_size(); i++) {
    EXPECT_FLOAT_EQ((*expected)[i], (*result_ptr)[i]);
  }
}