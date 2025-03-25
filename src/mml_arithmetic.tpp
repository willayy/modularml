#pragma once

#include "mml_arithmetic.hpp"
#include <memory>
#include <cmath> // For exp()
#include "mml_tensor.hpp"

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
std::shared_ptr<Tensor<T>> Arithmetic_mml<T>::reduce_max(
    const std::shared_ptr<const Tensor<T>> input, int axis) const {
    
    if (!input) {
        throw std::invalid_argument("Input tensor cannot be null.");
    }

    auto input_mml = std::dynamic_pointer_cast<const Tensor_mml<T>>(input);
    if (!input_mml) {
        throw std::invalid_argument("reduce_max only supports Tensor_mml<T>.");
    }

    auto shape = input->get_shape();
    if (axis < 0) axis += shape.size();
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        throw std::invalid_argument("Invalid axis for reduce_max.");
    }

    // Create the reduced shape
    array_mml<int> reduced_shape = shape;
    reduced_shape[axis] = 1;

    auto output = std::make_shared<Tensor_mml<T>>(reduced_shape);
    output->fill(std::numeric_limits<T>::lowest());

    array_mml<int> idx(std::vector<int>(shape.size(), 0));
    size_t total_size = input->get_size();

    for (size_t i = 0; i < total_size; ++i) {
        // Access using multi-dimensional index and iterate it manually
        T value = (*input_mml)[idx];

        array_mml<int> reduced_idx = idx;
        reduced_idx[axis] = 0;

        T& out_val = (*output)[reduced_idx];
        out_val = std::max(out_val, value);

        // Increment index (multi dimensional counter)
        for (int d = shape.size() - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }

    return output;
}

template <typename T>
std::shared_ptr<Tensor<T>> Arithmetic_mml<T>::reduce_sum(
    const std::shared_ptr<const Tensor<T>> input, int axis) const {

    if (!input) {
        throw std::invalid_argument("Input tensor cannot be null.");
    }

    auto shape = input->get_shape();
    if (axis < 0) axis += shape.size();
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        throw std::runtime_error("Invalid axis for reduce_sum.");
    }

    auto input_mml = std::dynamic_pointer_cast<const Tensor_mml<T>>(input);
    if (!input_mml) {
        throw std::invalid_argument("reduce_sum only supports Tensor_mml<T>.");
    }

    array_mml<int> reduced_shape = shape;
    reduced_shape[axis] = 1;
    auto output = std::make_shared<Tensor_mml<T>>(reduced_shape);
    output->fill(0);

    // Calculate stride along the axis
    size_t stride = 1;
    for (int i = axis + 1; i < shape.size(); ++i) {
        stride *= shape[i];
    }

    int axis_dim = shape[axis];
    size_t total = input_mml->get_size();
    size_t group = total / axis_dim;

    for (size_t i = 0; i < group; ++i) {
        size_t base = (i / stride) * stride * axis_dim + (i % stride);
        T sum = 0;
        for (int j = 0; j < axis_dim; ++j) {
            sum += (*input_mml)[base + j * stride];
        }
        (*output)[i] = sum;
    }

    return output;
}
