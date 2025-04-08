#pragma once

#include "a_arithmetic_module.hpp"

template <typename T>
class Arithmetic_mml : public ArithmeticModule<T> {
 public:
  [[deprecated("Use TensorOperationsModule instead")]]
  Arithmetic_mml();
  [[deprecated("Use TensorOperationsModule instead")]]
  Arithmetic_mml(Arithmetic_mml&&) noexcept;
  [[deprecated("Use TensorOperationsModule instead")]]
  Arithmetic_mml(const Arithmetic_mml&);
  [[deprecated("Use TensorOperationsModule instead")]]
  ~Arithmetic_mml() override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void add(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void subtract(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void multiply(const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c) const override;
  [[deprecated("Use TensorOperationsModule instead")]]
  bool equals(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b) const override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void elementwise(const shared_ptr<const Tensor<T>> a, std::function<T(T)> f, const shared_ptr<Tensor<T>> c) const override;
  [[deprecated("Use TensorOperationsModule instead")]]
  int arg_max(const shared_ptr<const Tensor<T>> a) const override;
  [[deprecated("Use TensorOperationsModule instead")]]
  void elementwise_in_place(const shared_ptr<Tensor<T>> a, std::function<T(T)> f) const override;
};

#include "../backend/mml_arithmetic.tpp"