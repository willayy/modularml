#include <cassert>
#include <modularml.hpp>

int main() {
  Tensor<float> t0 = tensor_mll({3, 3});                               // zero pointer
  Tensor<float> t1 = tensor_mll({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});  // data pointer

  const auto expected_shape = std::vector<int>{3, 3};
  assert(t0.get_shape() == expected_shape);
  assert(t1.get_shape() == expected_shape);

  Tensor<float> t_res = t0 + t1;
}