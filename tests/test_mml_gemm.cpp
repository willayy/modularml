#include <gtest/gtest.h>

#include <modularml>

TEST(test_mml_gemm, test_gemm) {
  const auto a = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const auto c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  
  const float alpha = 1;
  const float beta = 0;
  
  const auto d = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{40, 46, 94, 109});

  TensorOperations<float>::gemm(0, 0, 2, 2, 3, alpha, beta, a, 3, b, 2, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_check_matrix_match) {
  const auto a = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{4, 5, 6, 7});
  const auto c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  
  const float alpha = 1;
  const float beta = 0;

  ASSERT_THROW(
      TensorOperations<float>::gemm(0, 0, 2, 2, 3, alpha, beta, a, 3, b, 2, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_transpose) {
  auto a = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 3}, array_mml<int>{1, 2, 3, 4, 5, 6});
  auto b = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 3}, array_mml<int>{1, 2, 3, 4, 5, 6});
  auto c = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 2});

  const int alpha = 1;
  const int beta = 1;
  
  auto d = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 2}, array_mml<int>{14, 32, 32, 77});

  TensorOperations<int>::gemm(0, 1, 2, 2, 3, alpha, beta, a, 3, b, 2, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, gemm_128x128_float) {
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(16384, 16384, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(16384, 16384, 0, 100);

  auto a = std::make_shared<Tensor<float>>(array_mml<size_t>{128, 128}, a_data);
  auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{128, 128}, b_data);
  auto c = std::make_shared<Tensor<float>>(array_mml<size_t>{128, 128});

  TensorOperations<float>::gemm(0, 0, 128, 128, 128, 1, 0, a, 128, b, 128, c, 128);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_256x256_float) {
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(65536, 65536, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(65536, 65536, 0, 100);

  auto a = std::make_shared<Tensor<float>>(array_mml<size_t>{256, 256}, a_data);
  auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{256, 256}, b_data);
  auto c = std::make_shared<Tensor<float>>(array_mml<size_t>{256, 256});

  TensorOperations<float>::gemm(0, 0, 256, 256, 256, 1, 0, a, 256, b, 256, c, 256);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_122x122_float) {
  // Here we check that gemm still works when size is not divisiable by 8 just
  // in case
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(122 * 122, 122 * 122, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(122 * 122, 122 * 122, 0, 100);

  auto a = std::make_shared<Tensor<float>>(array_mml<size_t>{122, 122}, a_data);
  auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{122, 122}, b_data);
  auto c = std::make_shared<Tensor<float>>(array_mml<size_t>{122, 122});

  TensorOperations<float>::gemm(0, 0, 122, 122, 122, 1, 0, a, 122, b, 122, c, 122);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_250x250_float) {
  // Here we check that gemm still works when size is not divisiable by 8 just
  // in case
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(250 * 250, 250 * 250, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(250 * 250, 250 * 250, 0, 100);

  auto a = std::make_shared<Tensor<float>>(array_mml<size_t>{250, 250}, a_data);
  auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{250, 250}, b_data);
  auto c = std::make_shared<Tensor<float>>(array_mml<size_t>{250, 250});

  TensorOperations<float>::gemm(0, 0, 250, 250, 250, 1, 0, a, 250, b, 250, c, 250);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_1000x1000_float) {
  // Here we check that gemm still works when size is not divisiable by 8 just
  // in case
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(1000 * 1000, 1000 * 1000, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(1000 * 1000, 1000 * 1000, 0, 100);

  auto a = std::make_shared<Tensor<float>>(array_mml<size_t>{1000, 1000}, a_data);
  auto b = std::make_shared<Tensor<float>>(array_mml<size_t>{1000, 1000}, b_data);
  auto c = std::make_shared<Tensor<float>>(array_mml<size_t>{1000, 1000});

  TensorOperations<float>::gemm(0, 0, 1000, 1000, 1000, 1, 0, a, 1000, b, 1000, c, 1000);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}
