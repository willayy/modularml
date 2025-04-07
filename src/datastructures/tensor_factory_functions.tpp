#pragma once
#include "datastructures/tensor_factory_functions.hpp"

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_constructor_1(const array_mml<uli> &dims, const array_mml<T> &values) {
  auto tensor = Tensor_mml<T>(dims, values);
  std::shared_ptr<Tensor<T>> ptr = std::make_shared<Tensor_mml<T>>(tensor);
  return ptr;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_constructor_2(const array_mml<uli> &dims) {
  auto tensor = Tensor_mml<T>(dims);
  std::shared_ptr<Tensor<T>> ptr = std::make_shared<Tensor_mml<T>>(tensor);
  return ptr;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_constructor_3(const std::initializer_list<uli> dims,
                  const std::initializer_list<T> values) {
  auto tensor = Tensor_mml<T>(dims, values);
  std::shared_ptr<Tensor<T>> ptr = std::make_shared<Tensor_mml<T>>(tensor);
  return ptr;
}

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_constructor_4(const std::initializer_list<uli> dims) {
  auto tensor = Tensor_mml<T>(dims);
  std::shared_ptr<Tensor<T>> ptr = std::make_shared<Tensor_mml<T>>(tensor);
  return ptr;
}