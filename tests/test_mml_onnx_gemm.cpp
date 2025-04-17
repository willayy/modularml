#include <gtest/gtest.h>

#include <modularml>

TEST(test_mml_onnx_gemm, test_inner_product_1) {
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
  TensorOperations::set_gemm_onnx_ptr<float>(
      mml_onnx_gemm_inner_product<float>);
  const std::shared_ptr<Tensor<float>> res =
      TensorOperations::gemm_onnx<float>(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_inner_product_2) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_onnx_ptr<float>(
      mml_onnx_gemm_inner_product<float>);
  const std::shared_ptr<Tensor<float>> res = TensorOperations::gemm_onnx<float>(a, b);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_outer_produt_1) {
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
  TensorOperations::set_gemm_onnx_ptr<float>(
      mml_onnx_gemm_outer_product<float>);
  const std::shared_ptr<Tensor<float>> res =
      TensorOperations::gemm_onnx<float>(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_outer_produt_2) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_onnx_ptr<float>(mml_onnx_gemm_outer_product<float>);
  const std::shared_ptr<Tensor<float>> res =
      TensorOperations::gemm_onnx<float>(a, b);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_row_wise_product_1) {
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
  TensorOperations::set_gemm_onnx_ptr<float>(
      mml_onnx_gemm_row_wise_product<float>);
  const std::shared_ptr<Tensor<float>> res =
      TensorOperations::gemm_onnx<float>(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_row_wise_product_2) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_onnx_ptr<float>(
      mml_onnx_gemm_row_wise_product<float>);
  const std::shared_ptr<Tensor<float>> res =
      TensorOperations::gemm_onnx<float>(a, b);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_col_wise_product_1) {
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
  TensorOperations::set_gemm_onnx_ptr<float>(
      mml_onnx_gemm_col_wise_product<float>);
  const std::shared_ptr<Tensor<float>> res =
      TensorOperations::gemm_onnx<float>(a, b, alpha, beta, 0, 0, c);
  ASSERT_EQ((*res), (*d));
}

TEST(test_mml_onnx_gemm, test_col_wise_product_2) {
  const std::shared_ptr<Tensor<float>> a =
      TensorFactory::create_tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::shared_ptr<Tensor<float>> b =
      TensorFactory::create_tensor<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const std::shared_ptr<Tensor<float>> d =
      TensorFactory::create_tensor<float>({2, 2}, {40, 46, 94, 109});
  TensorOperations::set_gemm_onnx_ptr<float>(
      mml_onnx_gemm_col_wise_product<float>);
  const std::shared_ptr<Tensor<float>> res =
      TensorOperations::gemm_onnx<float>(a, b);
  ASSERT_EQ((*res), (*d));
}