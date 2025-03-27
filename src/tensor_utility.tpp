#pragma once

#include <iostream>
#include <random>

#include "tensor_utility.hpp"

template <typename T>
bool tensors_are_close(Tensor<T> &t1, Tensor<T> &t2, T tolerance) {
  static_assert(
      std::is_arithmetic_v<T>,
      "Tensor type must be an arithmetic type (int, float, double, etc.).");

  if (t1.get_shape() != t2.get_shape()) {
    std::cerr << "Error: Tensors have different shapes and cannot be compared!"
              << std::endl;
    return false;
  }

  for (uli i = 0; i < t1.get_size(); i++) {
    T diff = std::abs(t1[i] - t2[i]);
    T tolerance_limit = std::max(0.00001f, std::abs(tolerance * (t2[i])));

    if (diff > tolerance_limit) {
      std::cerr << "Difference of " << diff << " found at (" << i
                << ") which is too large." << std::endl;
      std::cerr << "Tolerance limit is " << tolerance_limit << std::endl;
      return false;
    }
  }

  return true;
}

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

template <typename T>
static auto generate_random_tensor(const array_mml<uli> &shape, T lo_v,
                                   T hi_v) {
  static_assert(
      std::is_arithmetic_v<T>,
      "Tensor type must be an arithmetic type (int, float, double, etc.).");
  Tensor_mml<T> tensor(shape);
  std::random_device rd;
  std::mt19937 gen(rd());

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(lo_v, hi_v);
    for (size_t i = 0; i < tensor.get_size(); i++) {
      tensor[i] = dist(gen);
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(lo_v, hi_v);
    for (size_t i = 0; i < tensor.get_size(); i++) {
      tensor.operator[](i) = dist(gen);
    }
  }

  return move(tensor);
}

// External Random Number Generator Edition
template <typename T>
void kaimingUniform(shared_ptr<Tensor<T>> W, uli in_channels, uli kernel_size, std::mt19937& gen) {
  static_assert(std::is_floating_point_v<T>, "Kaiming Uniform initialization requires a floating-point type.");

  uli fan_in = in_channels * kernel_size * kernel_size;
  if (fan_in == 0) {
    throw std::invalid_argument("fan_in must be greater than zero.");
  }

  float limit = std::sqrt(6.0f / fan_in);
  std::uniform_real_distribution<T> dist(-limit, limit);

  for (size_t i = 0; i < W->get_size(); ++i) {
    (*W)[i] = dist(gen);
  }
}

// Internal Random Number Generator Edition
template <typename T>
void kaimingUniform(shared_ptr<Tensor<T>> W, uli in_channels, uli kernel_size) {
  std::random_device rd;
  std::mt19937 gen(rd());  // seeded automatically
  kaimingUniform(W, in_channels, kernel_size, gen);
}
