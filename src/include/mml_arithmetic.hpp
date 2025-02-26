#pragma once

#include "a_arithmetic_module.hpp"
#include "a_data_structure.hpp"
#include "globals.hpp"
#include "mml_vector.hpp"

class Arithmetic_mml : public ArithmeticModule<float> {
 public:
  ~Arithmetic_mml() override = default;

  // Override default constructor
  Arithmetic_mml() = default;

  // Override move constructor
  Arithmetic_mml(Arithmetic_mml&&) noexcept = default;

  // Override copy constructor
  Arithmetic_mml(const Arithmetic_mml&) = default;

  unique_ptr<DataStructure<float>> add(const unique_ptr<DataStructure<float>> a, const unique_ptr<DataStructure<float>> b) const override {
    const int size = a->get_size();
    auto res_raw = vector<float>(size, 0);
    for (int i = 0; i < size; i++) {
      res_raw[i] = a->get_elem(i) + b->get_elem(i);
    }
    return make_unique<Vector_mml<float>>(res_raw);
  }

  unique_ptr<DataStructure<float>> subtract(const unique_ptr<DataStructure<float>> a, const unique_ptr<DataStructure<float>> b) const override {
    const int size = a->get_size();
    auto res_raw = vector<float>(size, 0);
    for (int i = 0; i < size; i++) {
      res_raw[i] = a->get_elem(i) - b->get_elem(i);
    }
    return make_unique<Vector_mml<float>>(res_raw);
  }

#pragma GCC diagnostic ignored "-Wunused-parameter"
  unique_ptr<DataStructure<float>> multiply(const unique_ptr<DataStructure<float>> a, const unique_ptr<DataStructure<float>> b) const override {
    return nullptr;  // TODO: Implement this
  }

  unique_ptr<DataStructure<float>> multiply(const unique_ptr<DataStructure<float>> a, const float b) const override {
    const int size = a->get_size();
    auto res_raw = vector<float>(size, 0);
    for (int i = 0; i < size; i++) {
      res_raw[i] = a->get_elem(i) * b;
    }
    return make_unique<Vector_mml<float>>(res_raw);
  }

  unique_ptr<DataStructure<float>> divide(const unique_ptr<DataStructure<float>> a, const float b) const override {
    const int size = a->get_size();
    auto res_raw = vector<float>(size, 0);
    for (int i = 0; i < size; i++) {
      res_raw[i] = a->get_elem(i) / b;
    }
    return make_unique<Vector_mml<float>>(res_raw);
  }

  bool equals(const unique_ptr<DataStructure<float>> a, const unique_ptr<DataStructure<float>> b) const override {
    if (bool same_shape = a->get_size() == b->get_size(); !same_shape) {
      return false;
    }
    const int size = a->get_size();
    for (int i = 0; i < size; i++) {
      if (!f_eq(a->get_elem(i), b->get_elem(i))) {
        return false;
      }
    }
    return true;
  }

  unique_ptr<ArithmeticModule<float>> clone() const override {
    return make_unique<Arithmetic_mml>(*this);
  }

 private:
  bool f_eq(float a, float b) const {
    return std::abs(a - b) < 1e-2f;
  }

  bool d_eq(double a, double b) const {
    return std::abs(a - b) < 1e-2;
  }
};