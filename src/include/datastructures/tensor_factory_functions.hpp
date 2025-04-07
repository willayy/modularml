#pragma once
#include "mml_tensor.hpp"
#include "../utility/uli.hpp"
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
#include <vector>

template <typename T>
static std::shared_ptr<Tensor<T>> mml_constructor_1(const array_mml<uli> &dims,
                                                    const array_mml<T> &values);

template <typename T>
static std::shared_ptr<Tensor<T>> mml_constructor_2(const array_mml<uli> &dims);

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_constructor_3(const std::initializer_list<uli> dims,
                  const std::initializer_list<T> values);

template <typename T>
static std::shared_ptr<Tensor<T>>
mml_constructor_4(const std::initializer_list<uli> dims);

#include "../datastructures/tensor_factory_functions.tpp"