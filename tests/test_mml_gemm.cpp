#include <gtest/gtest.h>

#include <modularml>
#include <iostream>

const shared_ptr<GemmModule<float>> gm_f = make_shared<Gemm_mml<float>>();
const shared_ptr<GemmModule<int>> gm_i = make_shared<Gemm_mml<int>>();

TEST(test_mml_gemm, test_inner_product_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_outer_produt_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_outer_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_row_wise_product_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_row_wise_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_col_wise_product_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  gm_f->gemm_col_wise_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  ASSERT_EQ((*c), (*d));
}

TEST(test_mml_gemm, test_gemm_properties) {
  for (int i = 0; i < 100; i++) {
    array_mml<int> shape = generate_random_array_mml_integral<int>(2,2);
    shape[0] = shape[1];
    const auto elements = accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    array_mml<int> data1 = generate_random_array_mml_integral<int>(elements, elements);
    array_mml<int> data2 = generate_random_array_mml_integral<int>(elements, elements);
    array_mml<int> data3 = array_mml<int>(elements);
    shared_ptr<Tensor<int>> t1 = make_shared<Tensor_mml<int>>(shape, data1);
    shared_ptr<Tensor<int>> t2 = make_shared<Tensor_mml<int>>(shape, data2);
    shared_ptr<Tensor<int>> tc = make_shared<Tensor_mml<int>>(shape, data3);
    const int alpha = 1;
    const int beta = 0;

    gm_i->gemm_inner_product(0, 0, shape[0], shape[1], shape[1], alpha, t1, shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r1 = (*tc).copy();
    tc->fill(0);
    gm_i->gemm_outer_product(0, 0, shape[0], shape[1], shape[1], alpha, t1, shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r2 = (*tc).copy();
    tc->fill(0);
    gm_i->gemm_row_wise_product(0, 0, shape[0], shape[1], shape[1], alpha, t1, shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r3 = (*tc).copy();
    tc->fill(0);
    gm_i->gemm_col_wise_product(0, 0, shape[0], shape[1], shape[1], alpha, t1, shape[0], t2, shape[0], beta, tc, shape[0]);
    auto r4 = (*tc).copy();
    std::cout << (*r1) << std::endl;
    std::cout << (*r2) << std::endl;
    std::cout << (*r3) << std::endl;
    std::cout << (*r4) << std::endl;
    auto prop = ((*r1) == (*r2)) && ((*r2) == (*r3)) && ((*r3) == (*r4));
    ASSERT_TRUE(prop);
  }
}
