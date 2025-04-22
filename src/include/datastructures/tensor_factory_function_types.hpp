#pragma once
#include <functional>
#include <initializer_list>
#include <memory>

namespace tfft {

template <typename T>
using tensor_constructor_func_1 = std::function<std::shared_ptr<Tensor<T>>(
    const array_mml<size_t> &shape, const array_mml<T> &data)>;

template <typename T>
using tensor_constructor_func_2 =
    std::function<std::shared_ptr<Tensor<T>>(const array_mml<size_t> &shape)>;

template <typename T>
using tensor_constructor_func_3 = std::function<std::shared_ptr<Tensor<T>>(
    const std::initializer_list<size_t> shape,
    const std::initializer_list<T> data)>;

template <typename T>
using tensor_constructor_func_4 = std::function<std::shared_ptr<Tensor<T>>(
    const std::initializer_list<size_t> shape)>;
    
}  // namespace tfft