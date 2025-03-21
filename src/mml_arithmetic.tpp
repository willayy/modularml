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

    // Ensure the input tensor is valid
    if (!input) {
        throw std::invalid_argument("Input tensor cannot be null.");
    }

    auto shape = input->get_shape();

    // Adjust negative axis values
    if (axis < 0) {
        axis += shape.size();
    }

    // Validate axis range
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        throw std::runtime_error("Invalid axis for reduce_max.");
    }

    // Convert input to Tensor_mml
    auto input_mml = std::static_pointer_cast<const Tensor_mml<T>>(input);
    if (!input_mml) {
        throw std::invalid_argument("reduce_max only supports Tensor_mml<T>.");
    }

    // Compute the reduced shape by setting the reduced axis to 1
    array_mml<int> reduced_shape = shape;
    reduced_shape[axis] = 1;

    // Allocate memory for the output tensor
    size_t reduced_size = 1;
    for (int dim : reduced_shape) {
        reduced_size *= dim;
    }

    auto output = std::make_shared<Tensor_mml<T>>(reduced_shape);
    output->fill(std::numeric_limits<T>::lowest());  // Initialize with lowest possible value

    size_t input_size = input_mml->get_size();
    size_t output_size = output->get_size();

    // Iterate over the input tensor and perform max reduction
    for (size_t i = 0; i < input_size; ++i) {
        auto idx = input_mml->public_unflatten_index(i);
        array_mml<int> reduced_idx = idx;
        reduced_idx[axis] = 0;  // Reduce along the specified axis

        // Compute the corresponding index in the output tensor
        size_t out_index = output->public_index_with_offset(reduced_idx);

        // Check bounds to prevent out of range access
        if (out_index >= output_size) {
            throw std::runtime_error("reduce_max: Output index out of bounds.");
        }

        if (i >= input_mml->get_size()) {
            throw std::runtime_error("reduce_max: Input index out of bounds.");
        }

        // Perform max operation
        (*output)[out_index] = std::max((*output)[out_index], (*input_mml)[i]);
    }

    return output;
}

template <typename T>
std::shared_ptr<Tensor<T>> Arithmetic_mml<T>::reduce_sum(
    const std::shared_ptr<const Tensor<T>> input, int axis) const {
    
    // Ensure input tensor is valid
    if (!input) {
        throw std::invalid_argument("Input tensor cannot be null.");
    }

    auto shape = input->get_shape();

    // Adjust negative axis values
    if (axis < 0) {
        axis += shape.size();
    }

    // Validate axis range
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        throw std::runtime_error("Invalid axis for reduce_sum.");
    }

    // Convert input to Tensor_mml
    auto input_mml = std::dynamic_pointer_cast<const Tensor_mml<T>>(input);
    if (!input_mml) {
        throw std::invalid_argument("reduce_sum only supports Tensor_mml<T>.");
    }

    // Compute reduced shape by setting the reduced axis to 1
    array_mml<int> reduced_shape = shape;
    reduced_shape[axis] = 1;  

    // Allocate memory for the output tensor
    auto output = std::make_shared<Tensor_mml<T>>(reduced_shape);
    output->fill(0);  // Initialize with zero

    // Iterate over input tensor and perform sum reduction
    for (size_t i = 0; i < input_mml->get_size(); ++i) {
        auto idx = input_mml->public_unflatten_index(i);
        idx[axis] = 0;  // Reduce along the specified axis

        // Compute corresponding index in output tensor
        size_t out_index = output->public_index_with_offset(idx);

        // Perform summation
        (*output)[out_index] += (*input_mml)[i];
    }

    return output;
}

template <typename T>
std::shared_ptr<Tensor<T>> Arithmetic_mml<T>::elementwise_softmax(
    std::shared_ptr<const Tensor<T>> input, int axis) const {
    
    // Ensure input tensor is valid
    if (!input) {
        throw std::invalid_argument("Input tensor cannot be null.");
    }

    auto shape = input->get_shape();

    // Adjust negative axis values
    if (axis < 0) {
        axis += shape.size();
    }

    // Validate axis range
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        throw std::invalid_argument("Invalid axis for elementwise_softmax.");
    }

    // Allocate memory for output tensor
    auto output = std::make_shared<Tensor_mml<T>>(shape);

    // Convert input to Tensor_mml
    auto input_mml = std::dynamic_pointer_cast<const Tensor_mml<T>>(input);
    if (!input_mml) {
        throw std::invalid_argument("elementwise_softmax requires Tensor_mml<T>.");
    }

    // Compute max values for numerical stability
    auto max_vals = reduce_max(input, axis);
    auto max_vals_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(max_vals);
    if (!max_vals_mml) {
        throw std::invalid_argument("reduce_max did not return a Tensor_mml<T>.");
    }

    // Compute exp(input - max) for each element
    auto temp = std::make_shared<Tensor_mml<T>>(shape);
    size_t input_size = input->get_size();

    for (size_t i = 0; i < input_size; i++) {
        auto idx = input_mml->public_unflatten_index(i);
        idx[axis] = 0; // Align with max values
        size_t max_index = max_vals_mml->public_index_with_offset(idx);

        (*temp)[i] = std::exp((*input_mml)[i] - (*max_vals_mml)[max_index]);
    }

    // Compute sum(exp(input - max)) along axis
    auto sum_vals = reduce_sum(temp, axis);
    auto sum_vals_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(sum_vals);
    if (!sum_vals_mml) {
        throw std::invalid_argument("reduce_sum did not return a Tensor_mml<T>.");
    }

    // Compute final softmax values
    for (size_t i = 0; i < input_size; i++) {
        auto idx = input_mml->public_unflatten_index(i);
        idx[axis] = 0;
        size_t sum_index = sum_vals_mml->public_index_with_offset(idx);

        (*output)[i] = (*temp)[i] / (*sum_vals_mml)[sum_index];
    }

    return output;
}