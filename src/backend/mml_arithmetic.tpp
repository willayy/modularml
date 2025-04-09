#pragma once

#include "backend/mml_arithmetic.hpp"

template <typename T> Arithmetic_mml<T>::Arithmetic_mml() = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(Arithmetic_mml &&) noexcept = default;

template <typename T>
Arithmetic_mml<T>::Arithmetic_mml(const Arithmetic_mml &) = default;

template <typename T> Arithmetic_mml<T>::~Arithmetic_mml() = default;

template <typename T>
void Arithmetic_mml<T>::add(const std::shared_ptr<Tensor<T>> a,
                            const std::shared_ptr<Tensor<T>> b,
                            std::shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::subtract(const std::shared_ptr<Tensor<T>> a,
                                 const std::shared_ptr<Tensor<T>> b,
                                 std::shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
void Arithmetic_mml<T>::multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                                 std::shared_ptr<Tensor<T>> c) const {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
bool Arithmetic_mml<T>::equals(const std::shared_ptr<Tensor<T>> a,
                               const std::shared_ptr<Tensor<T>> b) const {
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
int Arithmetic_mml<T>::arg_max(const std::shared_ptr<const Tensor<T>> a) const {
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

template <typename T>
void Arithmetic_mml<T>::elementwise(const std::shared_ptr<const Tensor<T>> a,
                                    std::function<T(T)> f,
                                    const std::shared_ptr<Tensor<T>> c) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<size_t> indices(num_dimensions);
  for (size_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }
  const auto total_elements = a->get_size();

  for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply std::function `f` from `a` to `c`
    (*c)[indices] = f((*a)[indices]);

    // Increment indices
    size_t d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    } while (d-- > 0);
  }
}

template <typename T>
void Arithmetic_mml<T>::elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                             std::function<T(T)> f) const {
  const auto shape = a->get_shape();
  const auto num_dimensions = shape.size();

  array_mml<size_t> indices(num_dimensions);
  for (size_t i = 0; i < num_dimensions; ++i) {
    indices[i] = 0;
  }

  const auto total_elements = a->get_size();

  for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Apply the std::function `f` to the current element
    (*a)[indices] = f((*a)[indices]);

    // Increment indices like a multi-dimensional counter
    size_t d = num_dimensions - 1;
    do {
      if (++indices[d] < shape[d]) {
        break; // No carry needed, continue iteration
      }
      indices[d] = 0; // Carry over to the next dimension
    } while (d-- > 0);
  }
}

template <typename T>
using WindowOpFn = std::function<T(
    const std::vector<T>& windowValues,
    std::optional<int64_t>& outIndex)>;

template <typename T>
void sliding_window_operation(
    const std::shared_ptr<const Tensor<T>>& input,
    std::shared_ptr<Tensor<T>>& output,
    const std::optional<std::shared_ptr<Tensor<int64_t>>>& indices_out,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const std::vector<std::pair<int, int>>& pads,
    WindowOpFn<T> window_op
) {
    const auto& in_shape = input->get_shape();
    const auto& out_shape = output->get_shape();
    size_t total_rank = in_shape.size();
    size_t spatial_rank = kernel_shape.size();
    
    std::vector<uli> out_idx(total_rank, 0);
    
    std::function<void(size_t)> recurse = [&](size_t dim) {
        if (dim == total_rank) {
            std::vector<T> windowValues;
            std::optional<int64_t> selected_index;
            
            std::vector<int> kernel_pos(spatial_rank, 0);
            
            std::function<void(size_t)> kernel_recurse = [&](size_t kdim) {
                if (kdim == spatial_rank) {
                    std::vector<uli> in_idx(total_rank, 0);

                    in_idx[0] = out_idx[0];
                    in_idx[1] = out_idx[1];
                    bool valid = true;
                    for (size_t i = 0; i < spatial_rank; ++i) {
                        int start = static_cast<int>(out_idx[i + 2]) * strides[i] - pads[i].first;
                        int pos = start + kernel_pos[i] * dilations[i];
                        if (pos < 0 || pos >= static_cast<int>(in_shape[i + 2])) {
                            valid = false;
                            break;
                        }
                        in_idx[i + 2] = static_cast<uli>(pos);
                    }
                    if (valid) {
                        windowValues.push_back((*input)[in_idx]);
                    }
                    return;
                }
                
                for (int k = 0; k < kernel_shape[kdim]; ++k) {
                    kernel_pos[kdim] = k;
                    kernel_recurse(kdim + 1);
                }
            };
            kernel_recurse(0);
            
            T out_value = window_op(windowValues, selected_index);
            (*output)[out_idx] = out_value;
            if (indices_out.has_value() && selected_index.has_value()) {
                (*indices_out.value())[out_idx] = selected_index.value();
            }
            return;
        }

        for (uli i = 0; i < out_shape[dim]; ++i) {
            out_idx[dim] = i;
            recurse(dim + 1);
        }
    };
    
    recurse(0);
}
