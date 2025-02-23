#pragma once
#include "a_arithmetic_module.hpp"
#include "a_data_structure.hpp"
#include "mml_vector_ds.hpp"

class Arithmetic_mml : public ArithmeticModule<float> {
 public:
  ~Arithmetic_mml() override = default;

  DataStructure<float>* add(const DataStructure<float>* a, const DataStructure<float>* b) const override {
    const auto& a_raw = a->get_raw_data();
    const auto& b_raw = b->get_raw_data();
    auto res_raw = vec<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] + b_raw[i];
    }
    Vector_mml<float>* res = new Vector_mml<float>(a->get_shape(), res_raw);
    return res;
  }

  DataStructure<float>* subtract(const DataStructure<float>* a, const DataStructure<float>* b) const override {
    const auto& a_raw = a->get_raw_data();
    const auto& b_raw = b->get_raw_data();
    auto res_raw = vec<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] - b_raw[i];
    }
    Vector_mml<float>* res = new Vector_mml<float>(a->get_shape(), res_raw);
    return res;
  }

#pragma GCC diagnostic ignored "-Wunused-parameter"
  DataStructure<float>* multiply(const DataStructure<float>* a, const DataStructure<float>* b) const override {
    return nullptr;  // TODO: Implement this
  }

  DataStructure<float>* multiply(const DataStructure<float>* a, const float b) const override {
    const auto& a_raw = a->get_raw_data();
    auto res_raw = vec<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] * b;
    }
    Vector_mml<float>* res = new Vector_mml<float>(a->get_shape(), res_raw);
    return res;
  }

  DataStructure<float>* divide(const DataStructure<float>* a, const float b) const override {
    const auto& a_raw = a->get_raw_data();
    auto res_raw = vec<float>(a->get_data_size(), 0);
    for (int i = 0; i < a->get_data_size(); i++) {
      res_raw[i] = a_raw[i] / b;
    }
    Vector_mml<float>* res = new Vector_mml<float>(a->get_shape(), res_raw);
    return res;
  }

  bool equals(const DataStructure<float>* a, const DataStructure<float>* b) const override {
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

 private:
  bool f_eq(float a, float b) const {
    return std::abs(a - b) < 1e-2f;
  }

  bool d_eq(double a, double b) const {
    return std::abs(a - b) < 1e-2;
  }
};