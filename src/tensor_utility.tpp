#pragma once

#include <iostream>
#include <random>

#include "tensor_utility.hpp"

template <typename T>
bool tensors_are_close(Tensor<T>& t1, Tensor<T>& t2, T tolerance) {
  static_assert(std::is_arithmetic_v<T>, "Tensor type must be an arithmetic type (int, float, double, etc.).");

  if (t1.get_shape() != t2.get_shape()) {
    std::cerr << "Error: Tensors have different shapes and cannot be compared!" << std::endl;
    return false;
  }

  for (int i = 0; i < t1.get_size(); i++) {
    T diff = std::abs(t1[i] - t2[i]);
    T tolerance_limit = std::abs(tolerance * (t2[i]));

    if (diff > tolerance_limit) {
      std::cerr << "Difference of " << diff << " found at (" << i << ") which is too large." << std::endl;
      std::cerr << "Tolerance limit is " << tolerance_limit << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T>
array_mml<T> generate_random_array_mml_integral(uint64_t lo_sz, uint64_t hi_sz, T lo_v, T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  uint64_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_int_distribution<T> int_dist(lo_v, hi_v);
  for (int i = 0; i < n; i++) {
    arr[i] = int_dist(gen);
  }
  return arr;
}

template <typename T>
array_mml<T> generate_random_array_mml_real(uint64_t lo_sz, uint64_t hi_sz, T lo_v, T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  uint64_t n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_real_distribution<T> real_dist(lo_v, hi_v);
  for (int i = 0; i < n; i++) {
    arr[i] = real_dist(gen);
  }
  return arr;
}