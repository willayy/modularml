#pragma once

#include "tensor_factory.hpp"

template <typename T>
TensorFactory<T> &TensorFactory<T>::getInstance() {
  static TensorFactory instance;
  return instance;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory<T>::create_tensor(const array_mml<uli> &shape,
                                const array_mml<T> &data) const {
  return this->tensor_constructor_1(shape, data);
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory<T>::create_tensor(const array_mml<uli> &shape) const {
  return this->tensor_constructor_2(shape);
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory<T>::create_tensor(const initializer_list<uli> shape,
                                const initializer_list<T> data) const {
  return this->tensor_constructor_3(shape, data);
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory<T>::create_tensor(const initializer_list<uli> shape) const {
  return this->tensor_constructor_4(shape);
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory<T>::random_tensor(const array_mml<uli> &shape, T lo_v,
                                T hi_v) const {
  uli n = 1;

  for (uli i = 0; i < shape.size(); i++) {
    n *= shape_ulis[i];
  }

  if constexpr (std::is_integral_v<T>) {
    array_mml<T> data = random_array_mml_integral(n, n, lo_v, hi_v);
    return create_tensor(shape, data);
  } else if constexpr (std::is_floating_point_v<T>) {
    array_mml<T> data = random_array_mml_real(n, n, lo_v, hi_v);
    return create_tensor(shape, data);
  }

  return nullptr;
}

template <typename T>
array_mml<T> TensorFactory<T>::random_array_mml_integral(uli lo_sz, uli hi_sz,
                                                         T lo_v, T hi_v) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  uli n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_int_distribution<T> int_dist(lo_v, hi_v);

  for (uli i = 0; i < n; i++) {
    arr[i] = int_dist(gen);
  }

  return arr;
}

template <typename T>
array_mml<T> TensorFactory<T>::random_array_mml_real(uli lo_sz, uli hi_sz,
                                                     T lo_v, T hi_v) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  uli n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_real_distribution<T> real_dist(lo_v, hi_v);

  for (uli i = 0; i < n; i++) {
    arr[i] = real_dist(gen);
  }

  return arr;
}

template <typename T>
void TensorFactory<T>::set_tensor_constructor(string id,
                                              function<void()> constructor) {
  if (id == "tensor_constructor_1") {
    static_assert(std::is_same_v<decltype(constructor),
                                 void (*)(const array_mml<uli> &,
                                          const array_mml<T> &) const>,
                  "Function signature does not match tensor_constructor_1.");
    tensor_constructor_1 = constructor;
  } else if (id == "tensor_constructor_2") {
    static_assert(std::is_same_v<decltype(constructor),
                                 void (*)(const array_mml<uli> &) const>,
                  "Function signature does not match tensor_constructor_2.");
    tensor_constructor_2 = constructor;
  } else if (id == "tensor_constructor_3") {
    static_assert(std::is_same_v<decltype(constructor),
                                 void (*)(initializer_list<uli> &,
                                          initializer_list<T> &) const>,
                  "Function signature does not match tensor_constructor_3.");
    tensor_constructor_3 = constructor;
  } else if (id == "tensor_constructor_4") {
    static_assert(std::is_same_v<decltype(constructor),
                                 void (*)(initializer_list<uli> &) const>,
                  "Function signature does not match tensor_constructor_4.");
    tensor_constructor_4 = constructor;
  } else {
    throw invalid_argument("Invalid constructor id.");
  }
}