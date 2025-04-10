#include <gtest/gtest.h>

#include <modularml>

const std::shared_ptr<GemmModule<float>> gm_f =
    std::make_shared<Gemm_mml<float>>();
const std::shared_ptr<GemmModule<int>> gm_i = std::make_shared<Gemm_mml<int>>();

TEST(test_mml_gemm, test_inner_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_outer_produt_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_outer_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_row_wise_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_row_wise_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_col_wise_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_col_wise_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_check_lda_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 1, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_ldb_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 1, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_ldc_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 1),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_dimensions_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 0, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 0, 0, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 0, 0, 0, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_tensor_sizes_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 1, 2, 3, alpha, a, 3, b, 2, beta, c, 3),
      std::invalid_argument);
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 1, 3, alpha, a, 3, b, 2, beta, c, 3),
      std::invalid_argument);
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 2, 2, alpha, a, 3, b, 2, beta, c, 3),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_tensor_properties_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3, 1}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_matrix_match_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({2, 2}, {4, 5, 6, 7});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_check_C_dimensions_1) {
  const std::shared_ptr<Tensor<float>> a =
      tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c = tensor_mml_p<float>({3, 2});
  const float alpha = 1;
  const float beta = 0;
  ASSERT_THROW(
      gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_transpose_1) {
  std::shared_ptr<Tensor<int>> a =
      tensor_mml_p<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  std::shared_ptr<Tensor<int>> b =
      tensor_mml_p<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  std::shared_ptr<Tensor<int>> c = tensor_mml_p<int>({2, 2});
  const int alpha = 1;
  const int beta = 1;
  std::shared_ptr<Tensor<int>> d = tensor_mml_p<int>({2, 2}, {22, 28, 49, 64});
  gm_i->gemm_inner_product(0, 1, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_gemm_properties) {
  for (int i = 0; i < 100; i++) {
    array_mml<size_t> shape = generate_random_array_mml_integral<size_t>(2, 2);
    shape[0] = shape[1];
    const auto elements =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    array_mml<int> data1 =
        generate_random_array_mml_integral<int>(elements, elements);
    array_mml<int> data2 =
        generate_random_array_mml_integral<int>(elements, elements);
    array_mml<int> data3 = array_mml<int>(elements);
    std::shared_ptr<Tensor<int>> t1 =
        std::make_shared<Tensor_mml<int>>(shape, data1);
    std::shared_ptr<Tensor<int>> t2 =
        std::make_shared<Tensor_mml<int>>(shape, data2);
    std::shared_ptr<Tensor<int>> tc =
        std::make_shared<Tensor_mml<int>>(shape, data3);
    const int alpha = 1;
    const int beta = 0;

    gm_i->gemm_inner_product(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                             shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r1 = (*tc).copy();
    tc->fill(0);
    gm_i->gemm_outer_product(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                             shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r2 = (*tc).copy();
    tc->fill(0);
    gm_i->gemm_row_wise_product(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                                shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r3 = (*tc).copy();
    tc->fill(0);
    gm_i->gemm_col_wise_product(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                                shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r4 = (*tc).copy();

    auto prop = ((*r1) == (*r2)) && ((*r2) == (*r3)) && ((*r3) == (*r4));
    ASSERT_TRUE(prop);
  }
}

TEST(test_mml_gemm, gemm_128x128_float) {
  array_mml<float> a_data = generate_random_array_mml_real<float>(16384, 16384, 0, 100);
  array_mml<float> b_data = generate_random_array_mml_real<float>(16384, 16384, 0, 100);
  
  shared_ptr<Tensor<float>> a = TensorFactory::create_tensor<float>({128, 128}, a_data);
  shared_ptr<Tensor<float>> b = TensorFactory::create_tensor<float>({128, 128}, b_data);
  shared_ptr<Tensor<float>> c = TensorFactory::create_tensor<float>({128, 128});
  
  TensorOperationsModule::gemm<float>(0, 0, 128, 128, 128, 1, a, 128, b, 128, 0, c, 128);
  
  ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_256x256_float) {
  array_mml<float> a_data = generate_random_array_mml_real<float>(65536, 65536, 0, 100);
  array_mml<float> b_data = generate_random_array_mml_real<float>(65536, 65536, 0, 100);
  
  shared_ptr<Tensor<float>> a = TensorFactory::create_tensor<float>({256, 256}, a_data);
  shared_ptr<Tensor<float>> b = TensorFactory::create_tensor<float>({256, 256}, b_data);
  shared_ptr<Tensor<float>> c = TensorFactory::create_tensor<float>({256, 256});
  
  TensorOperationsModule::gemm<float>(0, 0, 256, 256, 256, 1, a, 256, b, 256, 0, c, 256);
  
  ASSERT_TRUE(1); // This test is here to be able to check the time it takes for different GEMM inplementations
}



