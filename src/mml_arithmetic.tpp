#pragma once

#include "mml_arithmetic.hpp"

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml() = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(Arithmetic_mml&&) noexcept = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(const Arithmetic_mml&) = default;

template <typename T>
Arithmetic_mml<T>::~Arithmetic_mml() = default;

template <typename T>
void Arithmetic_mml<T>::add(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (int i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::subtract(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (int i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::multiply(const shared_ptr<Tensor<T>> a, const T b, shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (int i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
bool Arithmetic_mml<T>::equals(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b) const {
  if (a->get_size() != b->get_size() || a->get_shape() != b->get_shape()) {
    return false;
  } else {
    const auto size = a->get_size();
    for (int i = 0; i < size; i++) {
      if ((*a)[i] != (*b)[i]) {
        return false;
      }
    }
    return true;
  }
}

template <typename T>
void Arithmetic_mml<T>::elementwise(const shared_ptr<const Tensor<T>> a, T (*f)(T), const shared_ptr<Tensor<T>> c) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<int> indices(num_dimensions);
  for (uint64_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = a->get_size();

  for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply function `f` from `a` to `c`
    (*c)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    for (int d = num_dimensions - 1; d >= 0; --d) {
      if (++indices[d] < shape[d]) {
        break;  // No carry needed, continue iteration
      }
      indices[d] = 0;  // Carry over to the next dimension
    }
  }
}

template <typename T>
void Arithmetic_mml<T>::elementwise_in_place(const shared_ptr<Tensor<T>> a, T (*f)(T)) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<int> indices(num_dimensions);
  for (uint64_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }

  const auto total_elements = a->get_size();

  for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply the function `f` to the current element
    (*a)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    for (int d = num_dimensions - 1; d >= 0; --d) {
      if (++indices[d] < shape[d]) {
        break;  // No carry needed, move to the next iteration
      }
      indices[d] = 0;  // Carry over to the next dimension
    }
  }
}

template <typename T>
void Arithmetic_mml<T>::reduce_max(const shared_ptr<const Tensor<T>> input,
                                   shared_ptr<Tensor<T>> output,
                                   int axis) const {
  auto input_mml = static_cast<const Tensor_mml<T>*>(input.get());
  if (!input_mml) {
    throw std::invalid_argument("reduce_max only supports Tensor_mml<T>.");
  }

  auto shape = input_mml->get_shape();
  if (axis < 0 || axis >= static_cast<int>(shape.size())) {
    throw std::invalid_argument("Invalid axis for reduce_max.");
  }

  auto reduced_shape = shape;
  reduced_shape[axis] = 1;
  output->reshape(reduced_shape);
  output->fill(std::numeric_limits<T>::lowest());

  const auto input_size = input_mml->get_size();
  if (input_size == 0) {
    throw std::invalid_argument("Input tensor is empty.");
  }
  array_mml<int> idx; // Declare idx outside the loop
  for (size_t i = 0; i < input_size; ++i) {
    idx = input_mml->public_unflatten_index(i);
    idx[axis] = 0;
    size_t out_index = input_mml->public_index_with_offset(idx);
    (*output)[out_index] = std::max((*output)[out_index], (*input_mml)[i]);
  }
}

template <typename T>
void Arithmetic_mml<T>::reduce_sum(const shared_ptr<const Tensor<T>> input,
                                   shared_ptr<Tensor<T>> output,
                                   int axis) const {
  auto input_mml = static_cast<const Tensor_mml<T>*>(input.get());
  if (!input_mml) {
    throw std::invalid_argument("reduce_sum only supports Tensor_mml<T>.");
  }

  auto shape = input_mml->get_shape();
  if (axis < 0 || axis >= static_cast<int>(shape.size())) {
    throw std::invalid_argument("Invalid axis for reduce_sum.");
  }

  auto reduced_shape = shape;
  reduced_shape[axis] = 1;
  output->reshape(reduced_shape);
  output->fill(0);

  const auto input_size = input_mml->get_size();
  for (size_t i = 0; i < input_size; i++) {
    auto idx = input_mml->public_unflatten_index(i);
    idx[axis] = 0;
    size_t out_index = input_mml->public_index_with_offset(idx);
    (*output)[out_index] += (*input_mml)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::elementwise_softmax(
    const shared_ptr<const Tensor<T>> input, shared_ptr<Tensor<T>> output,
    int axis) const {
  auto input_mml = static_cast<const Tensor_mml<T>*>(input.get());
  if (!input_mml) {
    throw std::invalid_argument("elementwise_softmax requires Tensor_mml<T>.");
  }

  auto shape = input->get_shape();
  output->reshape(shape);
  auto reduced_shape = shape;
  reduced_shape[axis] = 1;

  auto max_vals = make_shared<Tensor_mml<T>>(reduced_shape);
  reduce_max(input, max_vals, axis);

  auto temp = std::make_shared<Tensor_mml<T>>(shape);
  const auto size = input->get_size();
  for (size_t i = 0; i < size; i++) {
    auto idx = input_mml->public_unflatten_index(i);
    idx[axis] = 0;
    size_t max_index = input_mml->public_index_with_offset(idx);
    (*temp)[i] = std::exp((*input_mml)[i] - (*max_vals)[max_index]);
  }

  auto sum_vals = make_shared<Tensor_mml<T>>(reduced_shape);
  this->reduce_sum(temp, sum_vals, axis);

  for (size_t i = 0; i < size; i++) {
    auto idx = input_mml->public_unflatten_index(i);
    idx[axis] = 0;
    size_t sum_index = input_mml->public_index_with_offset(idx);
    (*output)[i] = (*temp)[i] / (*sum_vals)[sum_index];
  }
}