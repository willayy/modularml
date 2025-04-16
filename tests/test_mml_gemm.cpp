#include <gtest/gtest.h>

#include <modularml>

TEST(test_mml_gemm, test_inner_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{40, 46, 94, 109});
  Gemm::inner_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_outer_produt_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{40, 46, 94, 109});
  Gemm::outer_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_row_wise_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{40, 46, 94, 109});
  Gemm::row_wise_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_col_wise_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{40, 46, 94, 109});
  Gemm::col_wise_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_check_lda_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 2, 3, alpha, a, 1, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_ldb_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 1, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_ldc_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 1),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_dimensions_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 0, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 0, 0, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 0, 0, 0, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_tensor_sizes_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{40, 46, 94, 109});
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 1, 2, 3, alpha, a, 3, b, 2, beta, c, 3),
      std::invalid_argument);
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 1, 3, alpha, a, 3, b, 2, beta, c, 3),
      std::invalid_argument);
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 2, 2, alpha, a, 3, b, 2, beta, c, 3),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_tensor_properties_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3, 1}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_matrix_match_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2}, array_mml<float>{4, 5, 6, 7});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_C_dimensions_1) {
  const std::shared_ptr<Tensor<float>> a =
      std::make_shared<Tensor<float>>(array_mml<size_t>{2, 3}, array_mml<float>{1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2}, array_mml<float>{4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{3, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      Gemm::inner_product<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_transpose_1) {
  std::shared_ptr<Tensor<int>> a =
      std::make_shared<Tensor<int>>(array_mml<size_t>{2, 3}, array_mml<int>{1, 2, 3, 4, 5, 6});
  std::shared_ptr<Tensor<int>> b =
      std::make_shared<Tensor<int>>(array_mml<size_t>{2, 3}, array_mml<int>{1, 2, 3, 4, 5, 6});
  std::shared_ptr<Tensor<int>> c = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 2});
  const int alpha = 1;
  const int beta = 1;
  std::shared_ptr<Tensor<int>> d = std::make_shared<Tensor<int>>(array_mml<size_t>{2, 2}, array_mml<int>{22, 28, 49, 64});
  Gemm::inner_product<int>(0, 1, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_gemm_properties) {
  for (int i = 0; i < 100; i++) {
    array_mml<size_t> shape = ArrayUtils::generate_random_array_mml_integral<size_t>(2, 2);
    shape[0] = shape[1];
    const auto elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data1 =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    array_mml<int> data2 =
        ArrayUtils::generate_random_array_mml_integral<int>(elements, elements);
    array_mml<int> data3 = array_mml<int>(elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor<int>>(shape, data1);
    std::shared_ptr<Tensor<int>> t2 =
        std::make_shared<Tensor<int>>(shape, data2);
    std::shared_ptr<Tensor<int>> tc =
        std::make_shared<Tensor<int>>(shape, data3);
    const int alpha = 1;
    const int beta = 0;

    Gemm::inner_product<int>(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                             shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r1 = (*tc).copy();
    tc->fill(0);
    Gemm::outer_product<int>(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                             shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r2 = (*tc).copy();
    tc->fill(0);
    Gemm::row_wise_product<int>(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                                shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r3 = (*tc).copy();
    tc->fill(0);
    Gemm::col_wise_product<int>(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                                shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r4 = (*tc).copy();

    auto prop = ((*r1) == (*r2)) && ((*r2) == (*r3)) && ((*r3) == (*r4));
    ASSERT_TRUE(prop);
  }
}

TEST(test_mml_gemm, gemm_128x128_float) {
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(16384, 16384, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(16384, 16384, 0, 100);
  
  std::shared_ptr<Tensor<float>> a = std::make_shared<Tensor<float>>(array_mml<size_t>{128, 128}, a_data);
  std::shared_ptr<Tensor<float>> b = std::make_shared<Tensor<float>>(array_mml<size_t>{128, 128}, b_data);
  std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{128, 128});
  
  Gemm::inner_product<float>(0, 0, 128, 128, 128, 1, a, 128, b, 128, 0, c, 128);
  
  ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_256x256_float) {
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(65536, 65536, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(65536, 65536, 0, 100);
  
  std::shared_ptr<Tensor<float>> a = std::make_shared<Tensor<float>>(array_mml<size_t>{256, 256}, a_data);
  std::shared_ptr<Tensor<float>> b = std::make_shared<Tensor<float>>(array_mml<size_t>{256, 256}, b_data);
  std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{256, 256});
  
  Gemm::inner_product<float>(0, 0, 256, 256, 256, 1, a, 256, b, 256, 0, c, 256);
  
  ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_122x122_float) {
  // Here we check that gemm still works when size is not divisiable by 8 just in case
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(122 * 122, 122 * 122, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(122 * 122, 122 * 122, 0, 100);
  
  std::shared_ptr<Tensor<float>> a = std::make_shared<Tensor<float>>(array_mml<size_t>{122, 122}, a_data);
  std::shared_ptr<Tensor<float>> b = std::make_shared<Tensor<float>>(array_mml<size_t>{122, 122}, b_data);
  std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{122, 122});
  
  Gemm::inner_product<float>(0, 0, 122, 122, 122, 1, a, 122, b, 122, 0, c, 122);
  
  ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_250x250_float) {
  // Here we check that gemm still works when size is not divisiable by 8 just in case
  array_mml<float> a_data = ArrayUtils::generate_random_array_mml_real<float>(250 * 250, 250 * 250, 0, 100);
  array_mml<float> b_data = ArrayUtils::generate_random_array_mml_real<float>(250 * 250, 250 * 250, 0, 100);
  
  std::shared_ptr<Tensor<float>> a = std::make_shared<Tensor<float>>(array_mml<size_t>{250, 250}, a_data);
  std::shared_ptr<Tensor<float>> b = std::make_shared<Tensor<float>>(array_mml<size_t>{250, 250}, b_data);
  std::shared_ptr<Tensor<float>> c = std::make_shared<Tensor<float>>(array_mml<size_t>{250, 250});
  
  Gemm::inner_product<float>(0, 0, 250, 250, 250, 1, a, 250, b, 250, 0, c, 250);
  
  ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}



