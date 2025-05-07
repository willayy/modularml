#pragma once

#include "datastructures/tensor.hpp"

namespace ArrayUtils {

template <typename T>
array_mml<T> generate_random_array_mml_integral(size_t lo_sz = 1,
                                                size_t hi_sz = 5, T lo_v = 1,
                                                T hi_v = 10);

template <typename T>
array_mml<T> generate_random_array_mml_real(size_t lo_sz = 1, size_t hi_sz = 5,
                                            T lo_v = 1, T hi_v = 100);

}  // namespace ArrayUtils

#define _ARRAY_UTILS_REAL(DT)                                            \
  template array_mml<DT> ArrayUtils::generate_random_array_mml_real<DT>( \
      size_t, size_t, DT, DT);

#define _ARRAY_UTILS_INTEGER(DT)                                             \
  template array_mml<DT> ArrayUtils::generate_random_array_mml_integral<DT>( \
      size_t, size_t, DT, DT);
