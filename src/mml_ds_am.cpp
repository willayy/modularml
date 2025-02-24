#include "mml_ds_am.hpp"

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::add(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
  if (a->get_shape() != b->get_shape()) {
    throw std::invalid_argument(
        "The data structures does not have the same shape");
  }

  vec<int> shape = a->get_shape();
  MML_DataStructure<T>* result = new MML_DataStructure(shape);

  for (int i = 0; i < a->get_size(); ++i) {
    result->set(a->get(i) + b->get(i));
  }
  return result;
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::subtract(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
  if (a->get_shape() != b->get_shape()) {
    throw std::invalid_argument(
        "The data structures does not have the same shape");
  }

  vec<int> shape = a->get_shape();
  MML_DataStructure<T>* result = new MML_DataStructure(shape);

  for (int i = 0; i < a->get_size(); ++i) {
    result->set(a->get(i) - b->get(i));
  }
  return result;
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::multiply(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
  if (a->get_shape().size() != a->get_shape().size()) {
    throw std::invalid_argument(
        "The data structures does not have the same shape");
  }
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::multiply(MML_DataStructure<T>* a, const T b) const {
  if (a->get_size() == 0) {
    throw std::invalid_argument("The data structure isn't has no shape");
  }

  vec<int> shape = a->get_shape();
  MML_DataStructure<T>* result = new MML_DataStructure(shape);

  for (int i = 0; i < a->get_size(); ++i) {
    result->set(i, a->get(i) * b);
  }
}

template <typename T>
MML_DataStructure<T>* MML_DataStructure_AM<T>::divide(MML_DataStructure<T>* a, const T b) const {
  // TODO
}

template <typename T>
bool MML_DataStructure_AM<T>::equal(MML_DataStructure<T>* a, MML_DataStructure<T>* b) const {
  // TODO
}