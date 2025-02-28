#include <gtest/gtest.h>

#include <modularml>

const shared_ptr<ArithmeticModule<float>> am = make_shared<Arithmetic_mml<float>>();

// Test add
TEST(test_mml_arithmetic, test_add_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({2, 3}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 3}, {5, 7, 9, 11, 13, 15});
  am->add(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_add_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({3, 3}, {2, 4, 6, 8, 10, 12, 14, 16, 18});
  am->add(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_div_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const float b = 2;
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 3}, {2, 4, 6, 8, 10, 12});
  am->multiply(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_div_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const float b = 0.5;
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({3, 3}, {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5});
  am->multiply(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_mul_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({2, 3}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 3}, {-3, -3, -3, -3, -3, -3});
  am->subtract(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_mul_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  am->subtract(a, b, c);
  ASSERT_EQ((*c), (*d));
}