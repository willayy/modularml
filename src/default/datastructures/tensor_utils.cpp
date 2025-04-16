#include "datastructures/tensor.hpp"
#include "datastructures/tensor_utils.hpp"
#include <iostream>
#include <random>

namespace TensorUtils {

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

  for (size_t i = 0; i < t1.get_size(); i++) {
    // Handle different numeric types properly to avoid ambiguous abs()
    T diff;
    T tolerance_limit;
    
    if constexpr (std::is_unsigned_v<T>) {
      // For unsigned types, use direct subtraction with checks to avoid underflow
      diff = t1[i] > t2[i] ? t1[i] - t2[i] : t2[i] - t1[i];
      
      tolerance_limit = std::max(static_cast<T>(1), static_cast<T>(tolerance * t2[i]));
    } else if constexpr (std::is_floating_point_v<T>) {
      diff = std::abs(t1[i] - t2[i]);
      tolerance_limit = std::max(static_cast<T>(0.00001), std::abs(tolerance * t2[i]));
    } else {
      // Avoid underflow for signed types
      diff = std::abs(static_cast<long long>(t1[i]) - static_cast<long long>(t2[i]));
      tolerance_limit = std::max(static_cast<T>(0), static_cast<T>(std::abs(tolerance * t2[i])));
    }

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
Tensor<T> generate_random_tensor(const array_mml<size_t> &shape, T lo_v,
                                   T hi_v) {
  Tensor<T> tensor(shape);
  std::random_device rd;
  std::mt19937 gen(rd());

  if constexpr (std::is_integral_v<T>) {
    if constexpr (std::is_same_v<T, bool>) {
      std::bernoulli_distribution dist(0.5); // 50% chance of true/false
      for (size_t i = 0; i < tensor.get_size(); i++) {
        tensor.operator[](i) = dist(gen);
      }
    } else {
      std::uniform_int_distribution<T> dist(lo_v, hi_v);
      for (size_t i = 0; i < tensor.get_size(); i++) {
        tensor.operator[](i) = dist(gen);
      }
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(lo_v, hi_v);
    for (size_t i = 0; i < tensor.get_size(); i++) {
      tensor.operator[](i) = dist(gen);
    }
  }

  return std::move(tensor);
}

// External Random Number Generator Edition
template <typename T>
void kaiming_uniform(std::shared_ptr<Tensor<T>> W, size_t in_channels,
                     size_t kernel_size, std::mt19937 &gen) {
  static_assert(
      std::is_floating_point_v<T>,
      "Kaiming Uniform initialization requires a floating-point type.");

  size_t fan_in = in_channels * kernel_size * kernel_size;
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
void kaiming_uniform(std::shared_ptr<Tensor<T>> W, size_t in_channels,
                     size_t kernel_size) {
  std::random_device rd;
  std::mt19937 gen(rd()); // seeded automatically
  kaiming_uniform(W, in_channels, kernel_size, gen);
}

}

#define TYPE(DT) _TENSOR_UTILS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE

#define TYPE(DT) _TENSOR_UTILS_REAL(DT)
#include "types_real.txt"
#undef TYPE