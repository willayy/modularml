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

  void elementwise(const shared_ptr<Tensor<T>> a, T (*f)(T), const shared_ptr<Tensor<T>> b) const override {
    const auto rows = a->get_shape()[0];
    const auto cols = a->get_shape()[1];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        (*b)[{i, j}] = f((*a)[{i, j}]);
      }
    }
  }

  void elementwise_in_place(const shared_ptr<Tensor<T>> a, T (*f)(T)) const override {
    const auto rows = a->get_shape()[0];
    const auto cols = a->get_shape()[1];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        (*a)[{i, j}] = f((*a)[{i, j}]);
      }
    }
  }

  shared_ptr<ArithmeticModule<T>> clone() const override {
    return make_shared<Arithmetic_mml>(*this);
  }
};