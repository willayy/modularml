#pragma once
#include "datastructures/tensor_factory.hpp"

template <typename T>
shared_ptr<Tensor<T>> (*TensorFactory::tensor_constructor_1)(
    const array_mml<uli> &shape, const array_mml<T> &data) = mml_constructor_1;

template <typename T>
shared_ptr<Tensor<T>> (*TensorFactory::tensor_constructor_2)(
    const array_mml<uli> &shape) = mml_constructor_2;

template <typename T>
shared_ptr<Tensor<T>> (*TensorFactory::tensor_constructor_3)(
    const initializer_list<uli> shape, const initializer_list<T> data) =
    mml_constructor_3;

template <typename T>
shared_ptr<Tensor<T>> (*TensorFactory::tensor_constructor_4)(
    const initializer_list<uli> shape) = mml_constructor_4;

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<uli> &shape,
                                const array_mml<T> &data) {
  return tensor_constructor_1<T>(shape, data);
}

template <typename T>
void TensorFactory::set_tensor_constructor_1(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const array_mml<uli> &shape,
                                                const array_mml<T> &data)) {
  tensor_constructor_1<T> = tensor_constructor;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const array_mml<uli> &shape) {
  return tensor_constructor_2<T>(shape);
}

template <typename T>
void TensorFactory::set_tensor_constructor_2(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const array_mml<uli> &shape)) {
  tensor_constructor_2<T> = tensor_constructor;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const initializer_list<uli> shape,
                                const initializer_list<T> data) {
  return tensor_constructor_3<T>(shape, data);
}

template <typename T>
void TensorFactory::set_tensor_constructor_3(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const initializer_list<uli> shape,
                                                const initializer_list<T> data)) {
  tensor_constructor_3<T> = tensor_constructor;
}

template <typename T>
shared_ptr<Tensor<T>>
TensorFactory::create_tensor(const initializer_list<uli> shape) {
  return tensor_constructor_4<T>(shape);
}

template <typename T>
void TensorFactory::set_tensor_constructor_4(
    shared_ptr<Tensor<T>> (*tensor_constructor)(const initializer_list<uli> shape)) {
  tensor_constructor_4<T> = tensor_constructor;
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