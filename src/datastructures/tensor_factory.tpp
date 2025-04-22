#pragma once
#include "datastructures/tensor_factory.hpp"
#include "datastructures/array_utility.hpp"

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<size_t> &shape,
                             const array_mml<T> &data) {
  return tensor_constructor_1<T>(shape, data);
}

template <typename... Ts>
void TensorFactory::set_tensor_constructor_1(
    tfft::tensor_constructor_func_1<Ts>... tensor_constructor) {
  (..., (tensor_constructor_1<Ts> = tensor_constructor));
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<size_t> &shape) {
  return tensor_constructor_2<T>(shape);
}

template <typename... Ts>
void TensorFactory::set_tensor_constructor_2(
    tfft::tensor_constructor_func_2<Ts>... tensor_constructor) {
  (..., (tensor_constructor_2<Ts> = tensor_constructor));
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const std::initializer_list<size_t> shape,
                             const std::initializer_list<T> data) {
  return tensor_constructor_3<T>(shape, data);
}

template <typename... Ts>
void TensorFactory::set_tensor_constructor_3(
    tfft::tensor_constructor_func_3<Ts>... tensor_constructor) {
  (..., (tensor_constructor_3<Ts> = tensor_constructor));
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const std::initializer_list<size_t> shape) {
  return tensor_constructor_4<T>(shape);
}

template <typename... Ts>
void TensorFactory::set_tensor_constructor_4(
    tfft::tensor_constructor_func_4<Ts>... tensor_constructor) {
  (..., (tensor_constructor_4<Ts> = tensor_constructor));
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::random_tensor(const array_mml<size_t> &shape, T lo_v, T hi_v) {
  size_t n = 1;

  for (const auto &dim : shape) {
    n *= dim;
  }

  if constexpr (std::is_integral_v<T>) {
    array_mml<T> data = generate_random_array_mml_integral(n, n, lo_v, hi_v);
    return create_tensor(shape, data);
  } else if constexpr (std::is_floating_point_v<T>) {
    array_mml<T> data = generate_random_array_mml_real(n, n, lo_v, hi_v);
    return create_tensor(shape, data);
  }

  return nullptr;
}