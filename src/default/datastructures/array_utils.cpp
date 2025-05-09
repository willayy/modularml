#include "datastructures/array_utils.hpp"

namespace ArrayUtils {

template <typename T>
array_mml<T> generate_random_array_mml_integral(size_t lo_sz, size_t hi_sz,
                                                T lo_v, T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  size_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);

  if constexpr (std::is_same_v<T, bool>) {
    std::bernoulli_distribution bool_dist(0.5);  // 50% chance of true/false
    for (size_t i = 0; i < n; i++) {
      arr[i] = bool_dist(gen);
    }
  } else {
    std::uniform_int_distribution<T> int_dist(lo_v, hi_v);
    for (size_t i = 0; i < n; i++) {
      arr[i] = int_dist(gen);
    }
  }

  return arr;
}

template <typename T>
array_mml<T> generate_random_array_mml_real(size_t lo_sz, size_t hi_sz, T lo_v,
                                            T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  size_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_real_distribution<T> real_dist(lo_v, hi_v);
  for (size_t i = 0; i < n; i++) {
    arr[i] = real_dist(gen);
  }
  return arr;
}

}  // namespace ArrayUtils

#define TYPE(DT) _ARRAY_UTILS_REAL(DT)
#include "types_real.txt"
#undef TYPE

#define TYPE(DT) _ARRAY_UTILS_INTEGER(DT)
#include "types_integer.txt"
#undef TYPE
