#pragma once
#include <functional>
#include <initializer_list>
#include <memory>

#include "tensor_concept.hpp"

namespace tfft {

template <TensorConcept::Types T>
using tensor_constructor_func_1 = std::function<std::shared_ptr<Tensor<T>>(
    const array_mml<size_t> &shape, const array_mml<T> &data)>;

template <TensorConcept::Types T>
using tensor_constructor_func_2 =
    std::function<std::shared_ptr<Tensor<T>>(const array_mml<size_t> &shape)>;

template <TensorConcept::Types T>
using tensor_constructor_func_3 = std::function<std::shared_ptr<Tensor<T>>(
    const std::initializer_list<size_t> shape,
    const std::initializer_list<T> data)>;

template <TensorConcept::Types T>
using tensor_constructor_func_4 = std::function<std::shared_ptr<Tensor<T>>(
    const std::initializer_list<size_t> shape)>;
    
}  // namespace tfft