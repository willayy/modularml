
#include <cassert>
#include <modularml>

const shared_ptr<GemmModule<float>> gm = make_shared<Gemm_mml<float>>();

void test_inner_product_1() {
  const shared_ptr<Tensor<float>> a = tensor_mml<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml<float>({2, 2}, {40, 46, 94, 109});
  gm->gemm_inner_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  assert((*c) == (*d));
}

void test_outer_produt_1() {
  const shared_ptr<Tensor<float>> a = tensor_mml<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml<float>({2, 2}, {40, 46, 94, 109});
  gm->gemm_outer_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  assert((*c) == (*d));
}

void test_row_wise_product_1() {
  const shared_ptr<Tensor<float>> a = tensor_mml<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml<float>({2, 2}, {40, 46, 94, 109});
  gm->gemm_row_wise_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  assert((*c) == (*d));
}

void test_col_wise_product_1() {
  const shared_ptr<Tensor<float>> a = tensor_mml<float>({2, 3}, {1, 2, 3, 4, 5, 6});
  const shared_ptr<Tensor<float>> b = tensor_mml<float>({3, 2}, {4, 5, 6, 7, 8, 9});
  const shared_ptr<Tensor<float>> c = tensor_mml<float>({2, 2});
  const float alpha = 1;
  const float beta = 0;
  const shared_ptr<Tensor<float>> d = tensor_mml<float>({2, 2}, {40, 46, 94, 109});
  gm->gemm_col_wise_product(0, 0, 2, 2, 3, alpha, a, 3, b, 2, beta, c, 2);
  assert((*c) == (*d));
}

int main() {
  test_inner_product_1();
  return 0;
}