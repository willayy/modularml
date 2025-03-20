#pragma once

#include "mml_arithmetic.hpp"

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml() = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(Arithmetic_mml&&) noexcept = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(const Arithmetic_mml&) = default;

template <typename T>
Arithmetic_mml<T>::~Arithmetic_mml() = default;

template <typename T>
void Arithmetic_mml<T>::add(const shared_ptr<const Tensor<T>> a, const shared_ptr<const Tensor<T>> b, shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (int i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::subtract(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (int i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::multiply(const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (int i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
bool Arithmetic_mml<T>::equals(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b) const {
  if (a->get_size() != b->get_size() || a->get_shape() != b->get_shape()) {
    return false;
  } else {
    const auto size = a->get_size();
    for (int i = 0; i < size; i++) {
      if ((*a)[i] != (*b)[i]) {
        return false;
      }
    }
    return true;
  }
}

template <typename T>
void Arithmetic_mml<T>::elementwise(const shared_ptr<const Tensor<T>> a, T (*f)(T), const shared_ptr<Tensor<T>> c) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<int> indices(num_dimensions);
  for (uint64_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = a->get_size();

  for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply function `f` from `a` to `c`
    (*c)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    for (int d = num_dimensions - 1; d >= 0; --d) {
      if (++indices[d] < shape[d]) {
        break;  // No carry needed, continue iteration
      }
      indices[d] = 0;  // Carry over to the next dimension
    }
  }
}

template <typename T>
void Arithmetic_mml<T>::elementwise_in_place(const shared_ptr<Tensor<T>> a, T (*f)(T)) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<int> indices(num_dimensions);
  for (uint64_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }

  const auto total_elements = a->get_size();

  for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply the function `f` to the current element
    (*a)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    for (int d = num_dimensions - 1; d >= 0; --d) {
      if (++indices[d] < shape[d]) {
        break;  // No carry needed, move to the next iteration
      }
      indices[d] = 0;  // Carry over to the next dimension
    }
  }
}