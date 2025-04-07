#pragma once

#include "backend/mml_arithmetic.hpp"

template <typename T> Arithmetic_mml<T>::Arithmetic_mml() = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(Arithmetic_mml &&) noexcept = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(const Arithmetic_mml &) = default;

template <typename T> Arithmetic_mml<T>::~Arithmetic_mml() = default;

template <typename T>
void Arithmetic_mml<T>::add(const std::shared_ptr<Tensor<T>> a,
                            const std::shared_ptr<Tensor<T>> b,
                            std::shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (uli i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::subtract(const std::shared_ptr<Tensor<T>> a,
                                 const std::shared_ptr<Tensor<T>> b,
                                 std::shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (uli i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                                 std::shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (uli i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
bool Arithmetic_mml<T>::equals(const std::shared_ptr<Tensor<T>> a,
                               const std::shared_ptr<Tensor<T>> b) const {
  if (a->get_size() != b->get_size() || a->get_shape() != b->get_shape()) {
    return false;
  } else {
    const auto size = a->get_size();
    for (uli i = 0; i < size; i++) {
      if ((*a)[i] != (*b)[i]) {
        return false;
      }
    }
    return true;
  }
}

template <typename T>
int Arithmetic_mml<T>::arg_max(const std::shared_ptr<const Tensor<T>> a) const {
  const auto size = a->get_size();
  if (size == 0) {
    throw std::runtime_error("arg_max called on an empty tensor.");
  }

  T max_value = (*a)[0];
  int max_index = 0;

  for (int i = 1; i < static_cast<int>(size); ++i) {
    if ((*a)[i] > max_value) {
      max_value = (*a)[i];
      max_index = i;
    }
  }

  return max_index;
}

template <typename T>
void Arithmetic_mml<T>::elementwise(const std::shared_ptr<const Tensor<T>> a,
                                    std::function<T(T)> f,
                                    const std::shared_ptr<Tensor<T>> c) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<uli> indices(num_dimensions);
  for (uli i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = a->get_size();

  for (uli linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply std::function `f` from `a` to `c`
    (*c)[indices] = f((*a)[indices]);

    // Increment indices
    uli d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    } while (d-- > 0);
  }
}

template <typename T>
void Arithmetic_mml<T>::elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                             std::function<T(T)> f) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<uli> indices(num_dimensions);
  for (uli i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }

  const auto total_elements = a->get_size();

  for (uli linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply the std::function `f` to the current element
    (*a)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    uli d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    } while (d-- > 0);
  }
}