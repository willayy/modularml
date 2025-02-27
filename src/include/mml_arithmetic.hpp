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
    const auto& a_raw = a->get_raw_data();
    const auto& b_raw = b->get_raw_data();
    auto res_raw = vector<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] + b_raw[i];
    }
    return make_unique<Vector_mml<float>>(a->get_shape(), res_raw);
  }

  unique_ptr<DataStructure<float>> subtract(const unique_ptr<DataStructure<float>> a, const unique_ptr<DataStructure<float>> b) const override {
    const auto& a_raw = a->get_raw_data();
    const auto& b_raw = b->get_raw_data();
    auto res_raw = vector<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] - b_raw[i];
    }
    return make_unique<Vector_mml<float>>(a->get_shape(), res_raw);
  }

#pragma GCC diagnostic ignored "-Wunused-parameter"
  unique_ptr<DataStructure<float>> multiply(const unique_ptr<DataStructure<float>> a, const unique_ptr<DataStructure<float>> b) const override {
    return nullptr;  // TODO: Implement this
  }

  unique_ptr<DataStructure<float>> multiply(const unique_ptr<DataStructure<float>> a, const float b) const override {
    const auto& a_raw = a->get_raw_data();
    auto res_raw = vector<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] * b;
    }
    return make_unique<Vector_mml<float>>(a->get_shape(), res_raw);
  }

  unique_ptr<DataStructure<float>> divide(const unique_ptr<DataStructure<float>> a, const float b) const override {
    const auto& a_raw = a->get_raw_data();
    auto res_raw = vector<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] / b;
    }
    return make_unique<Vector_mml<float>>(a->get_shape(), res_raw);
  }

  bool equals(const unique_ptr<DataStructure<float>> a, const unique_ptr<DataStructure<float>> b) const override {
    if (bool same_shape = a->get_shape() == b->get_shape(); !same_shape) {
      return false;
    }
    const auto& a_data = a->get_raw_data();
    const auto& b_data = b->get_raw_data();
    for (int i = 0; i < a->get_data_size(); i++) {
      if (!f_eq(a_data[i], b_data[i])) {
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