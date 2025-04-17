#include <gtest/gtest.h>

#include <modularml>

TEST(test_tensor_operations_module, test_add) {
  std::shared_ptr<const Tensor<int>> a =
      TensorFactory::create_tensor<int>({2}, {1, 2});
  std::shared_ptr<const Tensor<int>> b =
      TensorFactory::create_tensor<int>({2}, {3, 4});
  std::shared_ptr<Tensor<int>> c = TensorFactory::create_tensor<int>({2});
  TensorOperations::add<int>(a, b, c);
  std::shared_ptr<Tensor<int>> expected_c =
      TensorFactory::create_tensor({2}, {4, 6});
  ASSERT_EQ(*expected_c, *c);
}

TEST(test_tensor_operations_module, test_gemm) {
  std::shared_ptr<Tensor<int>> a =
      TensorFactory::create_tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  std::shared_ptr<Tensor<int>> b =
      TensorFactory::create_tensor<int>({3, 2}, {7, 8, 9, 10, 11, 12});
  std::shared_ptr<Tensor<int>> c = TensorFactory::create_tensor<int>({2, 2});
  TensorOperations::gemm<int>(0, 0, 2, 2, 3, 1, a, 3, b, 2, 0, c, 2);
  std::shared_ptr<Tensor<int>> expected_c =
      TensorFactory::create_tensor({2, 2}, {58, 64, 139, 154});
  ASSERT_EQ(*expected_c, *c);
}