#include <gtest/gtest.h>

#include <modularml>

const std::shared_ptr<ArithmeticModule<float>> am =
    std::make_shared<Arithmetic_mml<float>>();

// Test add
TEST(test_mml_arithmetic, test_add_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({2, 3}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 3}, {5, 7, 9, 11, 13, 15});
  am->add(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_add_2) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({3, 3}, {2, 4, 6, 8, 10, 12, 14, 16, 18});
  am->add(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_div_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const float b = 2;
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 3}, {2, 4, 6, 8, 10, 12});
  am->multiply(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_div_2) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const float b = 0.5;
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({3, 3}, {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5});
  am->multiply(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_mul_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({2, 3}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 3});
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 3}, {-3, -3, -3, -3, -3, -3});
  am->subtract(a, b, c);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_arithmetic, test_mul_2) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 3});
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  am->subtract(a, b, c);
  ASSERT_EQ((*c), (*d));
}

float square(float x) { return x * x; }
TEST(test_mml_arithmetic, test_elementwise) {
  const std::shared_ptr<Tensor<float>> a = tensor_mml_p<float>(
      {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  const std::shared_ptr<Tensor<float>> b = tensor_mml_p<float>(
      {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>(
      {3, 3}, {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f, 81.0f});
  am->elementwise(a, square, b);
  ASSERT_EQ(*b, *c);
}

TEST(test_mml_arithmetic, test_elementwise_in_place) {
  const std::shared_ptr<Tensor<float>> a = tensor_mml_p<float>(
      {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  const std::shared_ptr<Tensor<float>> b = tensor_mml_p<float>(
      {3, 3}, {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f, 81.0f});
  am->elementwise_in_place(a, square);
  ASSERT_EQ(*a, *b);
}

TEST(test_mml_arithmetic, test_elementwise_in_place_many_dimensions_4D) {
  const std::shared_ptr<Tensor<float>> a = tensor_mml_p<float>(
      {2, 2, 3, 2}, {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                     7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,

                     13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                     19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f});

  const std::shared_ptr<Tensor<float>> b = tensor_mml_p<float>(
      {2, 2, 3, 2}, {1.0f,   4.0f,   9.0f,   16.0f,  25.0f,  36.0f,
                     49.0f,  64.0f,  81.0f,  100.0f, 121.0f, 144.0f,

                     169.0f, 196.0f, 225.0f, 256.0f, 289.0f, 324.0f,
                     361.0f, 400.0f, 441.0f, 484.0f, 529.0f, 576.0f});

  am->elementwise_in_place(a, square);
  ASSERT_EQ(*a, *b);
}

TEST(test_mml_arithmetic, test_argmax_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  const int b = 5;
  ASSERT_EQ(am->arg_max(a), b);
}

TEST(test_mml_arithmetic, test_argmax_2) {
  const std::shared_ptr<Tensor<float>> a = tensor_mml_p<float>({0});
  ASSERT_THROW(am->arg_max(a), std::runtime_error);
}

TEST(test_mml_arithmetic, test_argmax_3) {
  const std::shared_ptr<Tensor<float>> a = tensor_mml_p<float>(
      {2, 3, 4}, {1.0f, 5.0f, 2.0f, 3.0f, 4.0f, 7.0f, 6.0f,  0.0f,
                  9.0f, 3.0f, 2.0f, 1.0f, 1.5f, 2.5f, 88.0f, 3.3f,
                  2.2f, 3.1f, 0.4f, 1.1f, 4.4f, 6.6f, 2.2f,  0.0f});
  const int expected_index = 14;
  ASSERT_EQ(am->arg_max(a), expected_index);
}

TEST(test_mml_arithmetic, test_argmax_4) {
  const std::shared_ptr<Tensor<float>> a = tensor_mml_p<float>(
      {2, 2, 3}, {-5.0f, -10.0f, -20.0f, -3.0f, -2.0f, -1.0f, -50.0f, -30.0f,
                  -60.0f, -7.0f, -8.0f, -9.0f});
  const int expected_index = 5;
  ASSERT_EQ(am->arg_max(a), expected_index);
}
