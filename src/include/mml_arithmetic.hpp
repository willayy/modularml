#pragma once
#include "a_arithmetic_module.hpp"
#include "a_data_structure.hpp"
#include "mml_vector_ds.hpp"

class Arithmetic_mml : public ArithmeticModule<float> {
 public:

  ~Arithmetic_mml() override = default;

  DataStructure<float>* add(DataStructure<float>* a, const DataStructure<float>* b) const override {
    for (int i = 0; i < a->get_size(); i++) {
      vec<int> ind = vec<int>{i};
      a->set(ind, a->get(ind) + b->get(ind));
    }
    return a;
  }

  DataStructure<float>* subtract(DataStructure<float>* a, const DataStructure<float>* b) const override {
    for (int i = 0; i < a->get_size(); i++) {
      vec<int> ind = vec<int>{i};
      a->set(ind, a->get(ind) - b->get(ind));
    }
    return a;
  }

#pragma GCC diagnostic ignored "-Wunused-parameter"
  DataStructure<float>* multiply(DataStructure<float>* a, const DataStructure<float>* b) const override {
    return nullptr; // TODO: Implement this
  }

  DataStructure<float>* multiply(DataStructure<float>* a, const float b) const override {
    for (int i = 0; i < a->get_size(); i++) {
      vec<int> ind = vec<int>{i};
      a->set(ind, a->get(ind) * b);
    }
    return a;
  }

  DataStructure<float>* divide(DataStructure<float>* a, const float b) const override {
    for (int i = 0; i < a->get_size(); i++) {
      vec<int> ind = vec<int>{i};
      a->set(ind, a->get(ind) / b);
    }
    return a;
  }

  bool equal(DataStructure<float>* a, const DataStructure<float>* b) const override {
    return a->equals(*b);
  }
};