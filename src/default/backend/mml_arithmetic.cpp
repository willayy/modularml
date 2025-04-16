#include "backend/mml_arithmetic.hpp"

template <typename T>
void Arithmetic::add(const std::shared_ptr<Tensor<T>> a,
                            const std::shared_ptr<Tensor<T>> b,
                            std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] + (*b)[i];
  }
}

template <typename T>
void Arithmetic::subtract(const std::shared_ptr<Tensor<T>> a,
                                 const std::shared_ptr<Tensor<T>> b,
                                 std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] - (*b)[i];
  }
}

template <typename T>
void Arithmetic::multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                                 std::shared_ptr<Tensor<T>> c) {
  const auto size = a->get_size();
  for (size_t i = 0; i < size; i++) {
    (*c)[i] = (*a)[i] * b;
  }
}

template <typename T>
bool Arithmetic::equals(const std::shared_ptr<Tensor<T>> a,
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
int Arithmetic::arg_max(const std::shared_ptr<const Tensor<T>> a) {
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
void Arithmetic::elementwise(const std::shared_ptr<const Tensor<T>> a,
                                    std::function<T(T)> f,
                                    const std::shared_ptr<Tensor<T>> c) {
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
void Arithmetic::elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                             std::function<T(T)> f) {
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

void Arithmetic::sliding_window(
    const array_mml<size_t>& in_shape,
    const array_mml<size_t>& out_shape,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& strides,
    const std::vector<int>& dilations,
    const std::vector<std::pair<int, int>>& pads,
    const std::function<void(const std::vector<std::vector<size_t>>&, const std::vector<size_t>&)> &window_f
  ) {
  size_t total_rank = in_shape.size();
  size_t spatial_rank = kernel_shape.size();
  
  std::vector<size_t> out_idx(total_rank, 0);
  
  std::function<void(size_t)> recurse = [&](size_t dim) {
      if (dim == total_rank) { // Depth reached

        std::vector<std::vector<size_t>> window_in_idx;
        std::vector<int> kernel_pos(spatial_rank, 0);
        
        std::function<void(size_t)> kernel_recurse = [&](size_t kdim) {
            if (kdim == spatial_rank) { // Depth reached
              bool valid = true;
              std::vector<size_t> in_idx(total_rank, 0);
              in_idx[0] = out_idx[0]; // Batch
              in_idx[1] = out_idx[1]; // Channel

              for (size_t i = 0; i < spatial_rank; ++i) {
                int out_coord = static_cast<int>(out_idx[i + 2]);
                int start = out_coord * strides[i] - pads[i].first;
                int offset = kernel_pos[i] * dilations[i];
                int pos = start + offset;

                if (pos < 0 || pos >= static_cast<int>(in_shape[i + 2])) {
                    valid = false;
                    break;
                }
                in_idx[i + 2] = static_cast<size_t>(pos);
              }

              if (valid) {
                window_in_idx.push_back(in_idx);
              }
              return;
            }
            
            for (int k = 0; k < kernel_shape[kdim]; ++k) {
              kernel_pos[kdim] = k;
              kernel_recurse(kdim + 1);
            }
        };
        kernel_recurse(0);

        window_f(window_in_idx, out_idx);
        return;
      }

      for (size_t i = 0; i < out_shape[dim]; ++i) {
        out_idx[dim] = i;
        recurse(dim + 1);
      }
  };
  
  recurse(0);
}

#define TYPE(DT) _ARITHMETIC(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE