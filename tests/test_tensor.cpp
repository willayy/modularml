#include <gtest/gtest.h>

#include <modularml>

// Test the default constructors
TEST(test_tensor, test_default_constructor_1) {
  std::shared_ptr<DataStructure<int>> data = std::make_shared<Vector_mml<int>>(5);
  Tensor<int> t1 = Tensor<int>(data, {5});
  std::vector<int> shape = std::vector<int>(1, 5);
  Tensor<int> t2 = Tensor<int>(data, shape);
  // These should be the same
  ASSERT_EQ(t1, t2);
}

// Test the copy constructor
TEST(test_tensor, test_copy_constructor_1) {
  std::shared_ptr<DataStructure<int>> data = std::make_shared<Vector_mml<int>>(5);
  Tensor<int> t1 = Tensor<int>(data, {5});
  Tensor<int> t2 = Tensor<int>(t1);
  // These should be the same
  ASSERT_EQ(t1, t2);
}

// Test the move constructor
TEST(test_tensor, test_move_constructor_1) {
  std::shared_ptr<DataStructure<int>> data = std::make_shared<Vector_mml<int>>(5);
  Tensor<int> source = Tensor<int>(data, {5});
  Tensor<int> dest = std::move(source);
  // Verify that the destination has taken ownership of the resources
  ASSERT_EQ(dest.get_shape(), std::vector<int>({5}));
}

TEST(test_tensor, test_reshape_1) {
  std::shared_ptr<DataStructure<int>> data = std::make_shared<Vector_mml<int>>(9);
  Tensor<int> t1 = Tensor<int>(data, {3, 3});
  t1.reshape({9});
  ASSERT_EQ(t1.get_shape(), std::vector<int>({9}));
}

TEST(test_tensor, test_indices_1) {
  std::shared_ptr<DataStructure<int>> data = std::make_shared<Vector_mml<int>>(9);
  Tensor<int> t1 = Tensor<int>(data, {3, 3});
  ASSERT_EQ((t1[{0, 0}]), 0);
  ASSERT_EQ((t1[{0, 1}]), 0);
  ASSERT_EQ((t1[{0, 2}]), 0);
  ASSERT_EQ((t1[{1, 0}]), 0);
  ASSERT_EQ((t1[{1, 1}]), 0);
  ASSERT_EQ((t1[{1, 2}]), 0);
  ASSERT_EQ((t1[{2, 0}]), 0);
  ASSERT_EQ((t1[{2, 1}]), 0);
  ASSERT_EQ((t1[{2, 2}]), 0);
  t1[{0, 0}] = 1;
  t1[{0, 1}] = 2;
  t1[{0, 2}] = 3;
  t1[{1, 0}] = 4;
  t1[{1, 1}] = 5;
  t1[{1, 2}] = 6;
  t1[{2, 0}] = 7;
  t1[{2, 1}] = 8;
  t1[{2, 2}] = 9;
  ASSERT_EQ((t1[{0, 0}]), 1);
  ASSERT_EQ((t1[{0, 1}]), 2);
  ASSERT_EQ((t1[{0, 2}]), 3);
  ASSERT_EQ((t1[{1, 0}]), 4);
  ASSERT_EQ((t1[{1, 1}]), 5);
  ASSERT_EQ((t1[{1, 2}]), 6);
  ASSERT_EQ((t1[{2, 0}]), 7);
  ASSERT_EQ((t1[{2, 1}]), 8);
  ASSERT_EQ((t1[{2, 2}]), 9);
}

TEST(test_tensor, test_index_1) {
  std::shared_ptr<DataStructure<int>> data = std::make_shared<Vector_mml<int>>(9);
  Tensor<int> t1 = Tensor<int>(data, {3, 3});
  ASSERT_EQ(t1[0], 0);
  ASSERT_EQ(t1[1], 0);
  ASSERT_EQ(t1[2], 0);
  ASSERT_EQ(t1[3], 0);
  ASSERT_EQ(t1[4], 0);
  ASSERT_EQ(t1[5], 0);
  ASSERT_EQ(t1[6], 0);
  ASSERT_EQ(t1[7], 0);
  ASSERT_EQ(t1[8], 0);
  t1[0] = 1;
  t1[1] = 2;
  t1[2] = 3;
  t1[3] = 4;
  t1[4] = 5;
  t1[5] = 6;
  t1[6] = 7;
  t1[7] = 8;
  t1[8] = 9;
  ASSERT_EQ(t1[0], 1);
  ASSERT_EQ(t1[1], 2);
  ASSERT_EQ(t1[2], 3);
  ASSERT_EQ(t1[3], 4);
  ASSERT_EQ(t1[4], 5);
  ASSERT_EQ(t1[5], 6);
  ASSERT_EQ(t1[6], 7);
  ASSERT_EQ(t1[7], 8);
  ASSERT_EQ(t1[8], 9);
}