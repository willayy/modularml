#pragma once
#include "datastructures/tensor_factory.hpp"

template <typename T>
std::function<std::shared_ptr<Tensor<T>>(const array_mml<uli> &shape,
                                         const array_mml<T> &data)>
    TensorFactory::tensor_constructor_1 = mml_constructor_1<T>;

template <typename T>
std::function<std::shared_ptr<Tensor<T>>(const array_mml<uli> &shape)>
    TensorFactory::tensor_constructor_2 = mml_constructor_2<T>;

template <typename T>
std::function<std::shared_ptr<Tensor<T>>(const std::initializer_list<uli> shape,
                                         const std::initializer_list<T> data)>
    TensorFactory::tensor_constructor_3 = mml_constructor_3<T>;

template <typename T>
std::function<std::shared_ptr<Tensor<T>>(
    const std::initializer_list<uli> shape)>
    TensorFactory::tensor_constructor_4 = mml_constructor_4<T>;

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<uli> &shape,
                             const array_mml<T> &data) {
  return tensor_constructor_1<T>(shape, data);
}

template <typename T>
void TensorFactory::set_tensor_constructor_1(
    const std::function<std::shared_ptr<Tensor<T>>(const array_mml<uli> &shape,
                                                   const array_mml<T> &data)>
        &tensor_constructor) {
  tensor_constructor_1<T> = tensor_constructor;
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<uli> &shape) {
  return tensor_constructor_2<T>(shape);
}

template <typename T>
void TensorFactory::set_tensor_constructor_2(
    const std::function<std::shared_ptr<Tensor<T>>(const array_mml<uli> &shape)>
        &tensor_constructor) {
  tensor_constructor_2<T> = tensor_constructor;
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const std::initializer_list<uli> shape,
                             const std::initializer_list<T> data) {
  return tensor_constructor_3<T>(shape, data);
}

template <typename T>
void TensorFactory::set_tensor_constructor_3(
    const std::function<std::shared_ptr<Tensor<T>>(
        const std::initializer_list<uli> shape,
        const std::initializer_list<T> data)> &tensor_constructor) {
  tensor_constructor_3<T> = tensor_constructor;
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const std::initializer_list<uli> shape) {
  return tensor_constructor_4<T>(shape);
}

template <typename T>
void TensorFactory::set_tensor_constructor_4(
    const std::function<std::shared_ptr<Tensor<T>>(
        const std::initializer_list<uli> shape)> &tensor_constructor) {
  tensor_constructor_4<T> = tensor_constructor;
}

template <typename T>
std::shared_ptr<Tensor<T>>
TensorFactory::random_tensor(const array_mml<uli> &shape, T lo_v, T hi_v) {
  uli n = 1;

  for (const auto &dim : shape) {
    n *= dim;
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