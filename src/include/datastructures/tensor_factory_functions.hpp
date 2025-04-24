#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>  // IWYU pragma: keep

#include "mml_tensor.hpp"
#include "tensor_concept.hpp"

template <TensorConcept::Types T>
static std::shared_ptr<Tensor<T>> mml_constructor_1(
    const array_mml<size_t> &dims, const array_mml<T> &values);

template <TensorConcept::Types T>
static std::shared_ptr<Tensor<T>> mml_constructor_2(
    const array_mml<size_t> &dims);

template <TensorConcept::Types T>
static std::shared_ptr<Tensor<T>> mml_constructor_3(
    const std::initializer_list<size_t> dims,
    const std::initializer_list<T> values);

template <TensorConcept::Types T>
static std::shared_ptr<Tensor<T>> mml_constructor_4(
    const std::initializer_list<size_t> dims);

#include "../datastructures/tensor_factory_functions.tpp"