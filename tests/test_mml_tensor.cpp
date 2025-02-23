#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <modularml>

#define assert_msg(name, condition)                         \
  if (!(condition)) {                                       \
    std::cerr << "Assertion failed: " << name << std::endl; \
  }                                                         \
  assert(condition);                                        \
  std::cout << name << ": " << (condition ? "Passed" : "Failed") << std::endl;

vec<float> random_vec_f(int size, float low, float max, int seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(low, max);
  vec<float> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = dist(gen);
  }
  return v;
}

vec<int> random_vec_i(int size, int low, int max, int seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution dist(low, max);
  vec<int> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = dist(gen);
  }
  return v;
}

int main() {
  Tensor<float> t0 = tensor_mll({3, 3});
  Tensor<float> t00 = tensor_mll({3, 3});
  Tensor<float> t1 = tensor_mll({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t2 = tensor_mll({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t3 = tensor_mll({3, 3}, {2, 4, 6, 8, 10, 12, 14, 16, 18});

  // Test equality operator
  assert_msg("Tensor equality test", t0 == t00)

  // Test inequality operator
  assert_msg("Tensor inequality test", t0 != t1)

  // Test correct shape
  const auto expected_shape = vec<int>{3, 3};
  assert_msg("Tensor shape test", t0.get_shape() == expected_shape)

  // Test adding two tensors
  Tensor<float> tt = t1 + t0;
  assert_msg("Tensor addition test", tt == t1)

  // Test subtracting two tensors
  tt = t1 - t0;
  assert_msg("Tensor subtraction test", tt == t1)

  // Test more subtraction
  tt = t1 - t2;
  assert_msg("Tensor subtraction test 2", tt == t0)

  // Test element wise multiplication
  tt = t2 * 2;
  assert_msg("Tensor element wise multiplication test", tt == t3)

  // Test element wise division
  tt = t3 / 2;
  assert_msg("Tensor element wise division test", tt == t2)

  // Test getting an element
  t1[{0, 0}] = 1.0;
  assert_msg("Tensor get/set element test 1", (t1[{0, 0}] == 1))
  t1[{0, 0}] = 10.0;
  assert_msg("Tensor get/set element test 2", (t1[{0, 0}] == 10))
  t1[{2, 2}] = 10.0;
  assert_msg("Tensor get/set element test 3", (t1[{2, 2}] == 10))
  t1[{1, 2}] = 10.0;
  assert_msg("Tensor get/set element test 4", (t1[{1, 2}] == 10))

  // property testing
  auto tensor_prop1 = [](Tensor<float> t) {
    assert_msg("Tensor property 1", t == (t * 2) / 2)
  };

  auto tensor_prop2 = [](Tensor<float> t) {
    assert_msg("Tensor property 2", t == (t + t) / 2)
  };

  auto tensor_prop3 = [](Tensor<float> t) {
    const auto t_temp = tensor_mll(t.get_shape());
    assert_msg("Tensor property 3", t_temp == (t - t))
  };

  auto tensor_prop4 = [](Tensor<float> t, int n) {
    Tensor<float> tp = tensor_mll(t.get_shape());
    for (int i = 0; i < (n - 1); i++) {
      tp = tp + t;
    }
    assert_msg("Tensor property 4", tp == t * (n-1))
  };

  auto tensor_prop5 = [](Tensor<float> t, int n) {
    assert_msg("Tensor property 5", t == t * n / n)
  };

  for (int i = 0; i < 100; i++) {
    std::mt19937 gen(i);
    std::uniform_int_distribution dist(0, 99);
    int rand_int = dist(gen);
    vec<int> shape = random_vec_i(5, 1, 10, i);
    int data_size = accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    vec<float> data = random_vec_f(data_size, 1, 10, i);
    Tensor<float> t = tensor_mll(shape, data);
    tensor_prop1(t);
    tensor_prop2(t);
    tensor_prop3(t);
    tensor_prop4(t, rand_int);
    tensor_prop5(t, rand_int);
  }

  std::cout << "All tests completed!" << std::endl;

  return 0;
}