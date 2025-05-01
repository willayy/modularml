#include "datastructures/tensor_operations.hpp"

template <typename T>
void TensorOperations<T>::add(const std::shared_ptr<const Tensor<T>> a, 
         const std::shared_ptr<const Tensor<T>> b,
         std::shared_ptr<Tensor<T>> c) {
          const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
void TensorOperations<T>::subtract(const std::shared_ptr<Tensor<T>> a,
              const std::shared_ptr<Tensor<T>> b,
              std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
void TensorOperations<T>::multiply(const std::shared_ptr<Tensor<T>> a, 
              const T b,
              std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
bool TensorOperations<T>::equals(const std::shared_ptr<Tensor<T>> a,
            const std::shared_ptr<Tensor<T>> b) {
  if (a->get_size() != b->get_size() || a->get_shape() != b->get_shape()) {
    return false;
  } else {
    const auto size = a->get_size();
    for (size_t i = 0; i < size; i++) {
      if ((*a)[i] != (*b)[i]) {
        return false;
      }
    }
    return true;
  }
}

template <typename T>
int TensorOperations<T>::arg_max(const std::shared_ptr<const Tensor<T>> a) {
  const auto size = a->get_size();
  if (size == 0) {
    throw std::runtime_error("arg_max called on an empty tensor.");
  }

  T max_value = (*a)[0];
  int max_index = 0;

  for (int i = 1; i < static_cast<int>(size); ++i) {
    if ((*a)[i] > max_value) {
      max_value = (*a)[i];
      max_index = i;
    }
  }

  return max_index;
}

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE