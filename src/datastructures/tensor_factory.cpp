#pragma once

#include "datastructures/tensor_factory.hpp"

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<uli> &shape,
                                const array_mml<T> &data) {
  return this->tensor_constructor_1(shape, data);
}

template <typename T>
void TensorFactory::set_tensor_constructor_1(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const array_mml<uli> &shape,
                                                const array_mml<T> &data)) {
  this->tensor_constructor_1 = tensor_constructor;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<uli> &shape) {
  return this->tensor_constructor_2(shape);
}

template <typename T>
void TensorFactory::set_tensor_constructor_2(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const array_mml<uli> &shape)) {
  this->tensor_constructor_2 = tensor_constructor;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const initializer_list<uli> shape,
                                const initializer_list<T> data) {
  return this->tensor_constructor_3(shape, data);
}

template <typename T>
void TensorFactory::set_tensor_constructor_3(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const initializer_list<uli> shape,
                                                const initializer_list<T> data)) {
  this->tensor_constructor_3 = tensor_constructor;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const initializer_list<uli> shape) {
  return this->tensor_constructor_4(shape);
}

template <typename T>
void TensorFactory::set_tensor_constructor_4(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const initializer_list<uli> shape)) {
  this->tensor_constructor_4 = tensor_constructor;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::random_tensor(const array_mml<uli> &shape, T lo_v,
                                T hi_v) {
  uli n = 1;

  for (uli i = 0; i < shape.size(); i++) {
    n *= shape[i];
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