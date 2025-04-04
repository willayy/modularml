#pragma once

#include <iostream>
#include <random>
#include "globals.hpp"

#include "datastructures/array_utility.hpp"

template <typename T>
array_mml<T> generate_random_array_mml_integral(uli lo_sz, uli hi_sz, T lo_v,
                                                T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  uli n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_int_distribution<T> int_dist(lo_v, hi_v);
  for (uli i = 0; i < n; i++) {
    arr[i] = int_dist(gen);
  }
  return arr;
}

template <typename T>
array_mml<T> generate_random_array_mml_real(uli lo_sz, uli hi_sz, T lo_v,
                                            T hi_v) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(lo_sz, hi_sz);
  uli n = size_dist(gen);
  array_mml<T> arr = array_mml<T>(n);
  std::uniform_real_distribution<T> real_dist(lo_v, hi_v);
  for (uli i = 0; i < n; i++) {
    arr[i] = real_dist(gen);
  }
  return arr;
}