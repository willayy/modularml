#include <modularml>
#include <gtest/gtest.h>

TEST(test_tensor_operations_module, test_add) {
    shared_ptr<const Tensor<int>> a = TensorFactory::create_tensor({2}, {1, 2});
    shared_ptr<const Tensor<int>> b = TensorFactory::create_tensor({2}, {3, 4});
    shared_ptr<Tensor<int>> c = TensorFactory::create_tensor({2});
    TensorOperationsModule::add(a, b, c);
    shared_ptr<Tensor<int>> expected_c = TensorFactory::create_tensor({2}, {4, 6});
    ASSERT_EQ(*expected_c, *c);
}