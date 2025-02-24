#pragma once

#include "a_arithmetic_module.hpp"
#include "mml_data_structure.hpp"

template <typename T>
class MML_DataStructure_AM : public ArithmeticModule<T, MML_DataStructure<T>> {
 public:
  MML_DataStructure<T>* add(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const override;

  MML_DataStructure<T>* subtract(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const override;

  MML_DataStructure<T>* multiply(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const override;

  MML_DataStructure<T>* multiply(MML_DataStructure<T>* a, const T b) const override;

  MML_DataStructure<T>* divide(MML_DataStructure<T>* a, const T b) const override;

  bool equal(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const override;
};