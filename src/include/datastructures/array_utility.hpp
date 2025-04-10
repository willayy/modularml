#pragma once

#include "datastructures/a_tensor.hpp"
#include "globals.hpp"

#define GENERATE_RANDOM_ARRAY_INTEGRAL(T)                                      \
  (std::is_integral_v<T>, "Random array generation (integral) requires an "    \
                          "integral type (int, long, etc.).");
template <typename T>
array_mml<T> generate_random_array_mml_integral(uli lo_sz = 1, uli hi_sz = 5,
                                                T lo_v = 1, T hi_v = 10);

#define GENERATE_RANDOM_ARRAY_REAL(T)                                          \
  (std::is_floating_point_v<T>, "Random array generation (real) requires a "   \
                                "floating-point type (float, double, etc.).");
template <typename T>
array_mml<T> generate_random_array_mml_real(uli lo_sz = 1, uli hi_sz = 5,
                                            T lo_v = 1, T hi_v = 100);

#include "../datastructures/array_utility.tpp"