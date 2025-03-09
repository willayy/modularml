#pragma once

#include "a_arithmetic_module.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"

template <typename T>
class Arithmetic_mml : public ArithmeticModule<T> {
 public:
  // Override default constructor
  Arithmetic_mml() = default;

  // Override move constructor
  Arithmetic_mml(Arithmetic_mml&&) noexcept = default;

  // Override copy constructor
  Arithmetic_mml(const Arithmetic_mml&) = default;

  // Override destructor
  ~Arithmetic_mml() override = default;

  void add(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const override {
    const auto size = a->get_size();
    for (int i = 0; i < size; i++) {
      (*c)[i] = (*a)[i] + (*b)[i];
    }
  }

  void subtract(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const override {
    const auto size = a->get_size();
    for (int i = 0; i < size; i++) {
      (*c)[i] = (*a)[i] - (*b)[i];
    }
  }

  void multiply(const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c) const override {
    const auto size = a->get_size();
    for (int i = 0; i < size; i++) {
      (*c)[i] = (*a)[i] * b;
    }
  }

  bool equals(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b) const override {
    if (a->get_size() != b->get_size() || a->get_shape() != b->get_shape()) {
      return false;
    } else {
      const auto size = a->get_size();
      for (int i = 0; i < size; i++) {
        if ((*a)[i] == (*b)[i]) {
          return false;
        }
      }
      return true;
    }
  }

  void elementwise(const shared_ptr<Tensor<T>> a, T (*f)(T), const shared_ptr<Tensor<T>> c) const override {
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

  void elementwise_in_place(const shared_ptr<Tensor<T>> a, T (*f)(T)) const override {
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
  shared_ptr<ArithmeticModule<T>> clone() const override {
    return make_shared<Arithmetic_mml>(*this);
  }
};