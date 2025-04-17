#include <gtest/gtest.h>

#include <modularml>

TEST(test_mml_gemm, test_inner_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_ptr<float>(mml_gemm_inner_product<float>);
  TensorOperations::gemm<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_outer_produt_1) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_ptr<float>(mml_gemm_outer_product<float>);
  TensorOperations::gemm<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_row_wise_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_ptr<float>(mml_gemm_row_wise_product<float>);
  TensorOperations::gemm<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_col_wise_product_1) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_ptr<float>(mml_gemm_col_wise_product<float>);
  TensorOperations::gemm<float>(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_check_matrix_match_1) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({2, 2}, {4, 5, 6, 7});
  const std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  TensorOperations::set_gemm_ptr<float>(mml_gemm_inner_product<float>);
  ASSERT_THROW(
      TensorOperations::gemm(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2),
      std::invalid_argument);
}

TEST(test_mml_gemm, test_transpose_1) {
  std::shared_ptr<Tensor<int>> a =
      TensorFactory::create_tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  std::shared_ptr<Tensor<int>> b =
      TensorFactory::create_tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  std::shared_ptr<Tensor<int>> c = TensorFactory::create_tensor<int>({2, 2});
  const int alpha = 1;
  const int beta = 1;
  std::shared_ptr<Tensor<int>> d =
      TensorFactory::create_tensor<int>({2, 2}, {22, 28, 49, 64});
  TensorOperations::set_gemm_ptr<int>(mml_gemm_inner_product<int>);
  TensorOperations::gemm(0, 1, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
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
        TensorFactory::create_tensor<int>(shape, data1);
    std::shared_ptr<Tensor<int>> t2 =
        TensorFactory::create_tensor<int>(shape, data2);
    std::shared_ptr<Tensor<int>> tc =
        TensorFactory::create_tensor<int>(shape, data3);
    const int alpha = 1;
    const int beta = 0;

    TensorOperations::set_gemm_ptr<int>(mml_gemm_inner_product<int>);
    TensorOperations::gemm(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                           shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r1 = (*tc).copy();
    tc->fill(0);

    TensorOperations::set_gemm_ptr<int>(mml_gemm_outer_product<int>);
    TensorOperations::gemm(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                           shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r2 = (*tc).copy();
    tc->fill(0);

    TensorOperations::set_gemm_ptr<int>(mml_gemm_row_wise_product<int>);
    TensorOperations::gemm(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                           shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r3 = (*tc).copy();
    tc->fill(0);

    TensorOperations::set_gemm_ptr<int>(mml_gemm_col_wise_product<int>);
    TensorOperations::gemm(0, 0, shape[0], shape[1], shape[1], alpha, t1,
                           shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r4 = (*tc).copy();

    auto prop = ((*r1) == (*r2)) && ((*r2) == (*r3)) && ((*r3) == (*r4));
    ASSERT_TRUE(prop);
  }
}

TEST(test_mml_gemm, gemm_128x128_float) {
  array_mml<float> a_data =
      generate_random_array_mml_real<float>(16384, 16384, 0, 100);
  array_mml<float> b_data =
      generate_random_array_mml_real<float>(16384, 16384, 0, 100);

  std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({128, 128}, a_data);
  std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({128, 128}, b_data);
  std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({128, 128});

  TensorOperations::gemm<float>(0, 0, 128, 128, 128, 1, a, 128, b, 128, 0, c,
                                128);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_256x256_float) {
  array_mml<float> a_data =
      generate_random_array_mml_real<float>(65536, 65536, 0, 100);
  array_mml<float> b_data =
      generate_random_array_mml_real<float>(65536, 65536, 0, 100);

  std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({256, 256}, a_data);
  std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({256, 256}, b_data);
  std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({256, 256});

  TensorOperations::gemm<float>(0, 0, 256, 256, 256, 1, a, 256, b, 256, 0, c,
                                256);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_122x122_float) {
  // Here we check that gemm still works when size is not divisiable by 8 just
  // in case
  array_mml<float> a_data =
      generate_random_array_mml_real<float>(122 * 122, 122 * 122, 0, 100);
  array_mml<float> b_data =
      generate_random_array_mml_real<float>(122 * 122, 122 * 122, 0, 100);

  std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({122, 122}, a_data);
  std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({122, 122}, b_data);
  std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({122, 122});

  TensorOperations::gemm<float>(0, 0, 122, 122, 122, 1, a, 122, b, 122, 0, c,
                                122);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}

TEST(test_mml_gemm, gemm_250x250_float) {
  // Here we check that gemm still works when size is not divisiable by 8 just
  // in case
  array_mml<float> a_data =
      generate_random_array_mml_real<float>(250 * 250, 250 * 250, 0, 100);
  array_mml<float> b_data =
      generate_random_array_mml_real<float>(250 * 250, 250 * 250, 0, 100);

  std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({250, 250}, a_data);
  std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({250, 250}, b_data);
  std::shared_ptr<Tensor<float>> c =
      TensorFactory::create_tensor<float>({250, 250});

  TensorOperations::gemm<float>(0, 0, 250, 250, 250, 1, a, 250, b, 250, 0, c,
                                250);

  ASSERT_TRUE(1);  // This test is here to be able to check the time it takes
                   // for different GEMM inplementations
}
