#pragma once

#include "a_arithmetic_module.hpp"

template <typename T>
class Arithmetic_mml : public ArithmeticModule<T> {
 public:
  Arithmetic_mml();

  Arithmetic_mml(Arithmetic_mml&&) noexcept;

  Arithmetic_mml(const Arithmetic_mml&);

  ~Arithmetic_mml() override;

  void add(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const override;

  void subtract(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const override;

  void multiply(const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c) const override;

  bool equals(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b) const override;
  
  void elementwise(const shared_ptr<const Tensor<T>> a, std::function<T(T)> f, const shared_ptr<Tensor<T>> c) const override;

  int arg_max(const shared_ptr<const Tensor<T>> a) const override;

  void elementwise_in_place(const shared_ptr<Tensor<T>> a, std::function<T(T)> f) const override;
};

#include "../backend/mml_arithmetic.tpp"