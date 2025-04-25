#include <gtest/gtest.h>

#include "nodes/transpose.hpp"

TEST(TransposeNode_test, test_forward) {
  // Define dimensions: M = 2, K = 3, N = 2.
  array_mml<size_t> shapeA({2, 3});
  array_mml<size_t> shapeY({3, 2});  // Output shape is [M, N]

  // Wrap each tensor in a shared pointer.
  auto A_ptr = TensorFactory::create_tensor<int>(shapeA, {1, 2, 3, 4, 5, 6});
  auto Y_ptr = TensorFactory::create_tensor<int>(shapeY);
  Y_ptr->fill(0.0f);  // Initialize Y to zero

  // Setup the iomap with tensor names
  std::string a_string = "A";
  std::string y_string = "Y";

  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[a_string] = A_ptr;
  iomap[y_string] = Y_ptr;

  // Attribute
  std::vector<int> perm = {1, 0};  // Reverse the axi

  TransposeNode node(a_string, y_string, perm);

  // Run the forward pass
  node.forward(iomap);

  // Retrieve the result from iomap
  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end())
      << "Output tensor Y not found in iomap after forward pass";

  // Extract the shared pointer to the output tensor
  auto result_ptr = std::get<std::shared_ptr<Tensor<int>>>(y_it->second);

  auto expected = TensorFactory::create_tensor<int>(shapeY, {1, 4, 2, 5, 3, 6});

  // Verify
  for (int i = 0; i < expected->get_size(); i++) {
    if (i >= expected->get_size() || i >= result_ptr->get_size()) {
      std::cerr << "Index out of bounds! i: " << i << std::endl;
    }

    EXPECT_EQ((*expected)[i], (*result_ptr)[i]);
  }
}

TEST(TransposeNode_test, test_forward_3d) {
  array_mml<size_t> shapeA({2, 3, 4});  // 3D input shape
  array_mml<size_t> shapeY({3, 2, 4});  // Output shape after transpose

  // Wrap each tensor in a shared pointer.
  auto A_ptr = TensorFactory::create_tensor<int>(
      shapeA, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,

               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  auto Y_ptr = TensorFactory::create_tensor<int>(shapeY);
  Y_ptr->fill(0);  // Initialize Y to zero

  // Setup the iomap with tensor names
  std::string a_string = "A";
  std::string y_string = "Y";

  std::unordered_map<std::string, GeneralDataTypes> iomap;
  iomap[a_string] = A_ptr;
  iomap[y_string] = Y_ptr;

  std::vector<int> perm = {1, 0, 2};  // Transpose the first two dimensions

  // Construct the TransposeNode
  TransposeNode node(a_string, y_string, perm);

  // Run the forward pass
  node.forward(iomap);

  // Retrieve the result from iomap
  auto y_it = iomap.find(y_string);
  ASSERT_NE(y_it, iomap.end())
      << "Output tensor Y not found in iomap after forward pass";

  // Extract the shared pointer to the output tensor
  auto result_ptr = std::get<std::shared_ptr<Tensor<int>>>(y_it->second);

  auto expected = TensorFactory::create_tensor<int>(shapeY);

  // Fill expected tensor by manually transposing A with perm {1, 0, 2}
  const auto& A = *A_ptr;
  auto& expected_tensor = *expected;

  for (size_t i = 0; i < shapeA[0]; ++i) {
    for (size_t j = 0; j < shapeA[1]; ++j) {
      for (size_t k = 0; k < shapeA[2]; ++k) {
        // Mapping original indices [i, j, k] to transposed indices
        size_t transposed_i = j;  // Swap 0th and 1st dimensions
        size_t transposed_j = i;
        size_t transposed_k = k;  // 2nd dimension stays the same
        size_t original_index = i * shapeA[1] * shapeA[2] + j * shapeA[2] + k;
        size_t transposed_index = transposed_i * shapeY[1] * shapeY[2] +
                                  transposed_j * shapeY[2] + transposed_k;

        expected_tensor[transposed_index] = A[original_index];
      }
    }
  }

  // Verify
  for (int i = 0; i < expected->get_size(); i++) {
    if (i >= expected->get_size() || i >= result_ptr->get_size()) {
      std::cerr << "Index out of bounds! i: " << i << std::endl;
    }
    EXPECT_EQ((*expected)[i], (*result_ptr)[i]);
  }
}
