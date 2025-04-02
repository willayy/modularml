#include <modularml>
#include <gtest/gtest.h>

TEST(test_tensor_operations_module, test_add) {
    shared_ptr<const Tensor<int>> a = tensor_mml_p({2}, {1, 2});
    shared_ptr<const Tensor<int>> b = tensor_mml_p({2}, {3, 4});
    shared_ptr<Tensor<int>> c = tensor_mml_p({2}, {0, 0});
    TensorOperationsModule::add(a, b, c);
    shared_ptr<Tensor<int>> expected_c = tensor_mml_p({2}, {4, 6});
    ASSERT_EQ(*expected_c, *c);
}