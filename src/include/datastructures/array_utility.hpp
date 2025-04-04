#pragma once

#include "datastructures/a_tensor.hpp"

#define GENERATE_RANDOM_ARRAY_INTEGRAL(T)                                      \
  (std::is_integral_v<T>, "Random array generation (integral) requires an "    \
                          "integral type (int, long, etc.).");
template <typename T>
array_mml<T> generate_random_array_mml_integral(size_t lo_sz = 1,
                                                size_t hi_sz = 5, T lo_v = 1,
                                                T hi_v = 10);

#define GENERATE_RANDOM_ARRAY_REAL(T)                                          \
  (std::is_floating_point_v<T>, "Random array generation (real) requires a "   \
                                "floating-point type (float, double, etc.).");
template <typename T>
array_mml<T> generate_random_array_mml_real(size_t lo_sz = 1, size_t hi_sz = 5,
                                            T lo_v = 1, T hi_v = 100);

                                            
template <typename T>
array_mml<T> generate_array_random_content(uli size, T lower_bound, T upper_bound);

template <typename T>
array_mml<T> generate_array_random_content_real(uli size, T lower_bound, T upper_bound);

#include "../datastructures/array_utility.tpp"