#include <gtest/gtest.h>

#include <modularml>

const shared_ptr<OnnxGemmModule<float>> gm_f = make_shared<OnnxGemm_mml<float>>();
const shared_ptr<OnnxGemmModule<int>> gm_i = make_shared<OnnxGemm_mml<int>>();

TEST(test_mml_onnx_gemm, test_inner_product_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_inner_product(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_inner_product_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_inner_product(a, b);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_outer_produt_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_outer_product(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_outer_produt_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_outer_product(a, b);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_row_wise_product_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_row_wise_product(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_row_wise_product_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_row_wise_product(a, b);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_col_wise_product_1) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml_p<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_col_wise_product(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_col_wise_product_2) {
  const shared_ptr<Tensor<float>> a = tensor_mml_p<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml_p<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> d = tensor_mml_p<float>({2, 2}, {40, 46, 94, 109});
  const shared_ptr<Tensor<float>> res = gm_f->gemm_col_wise_product(a, b);
  ASSERT_EQ((*res), (*d));
}