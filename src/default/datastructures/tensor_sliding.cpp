#include "datastructures/tensor_operations.hpp"

template <typename T>
void TensorOperations<T>::sliding_window(
    const array_mml<size_t> &in_shape, const array_mml<size_t> &out_shape,
    const std::vector<int> &kernel_shape, const std::vector<int> &strides,
    const std::vector<int> &dilations,
    const std::vector<std::pair<int, int>> &pads,
    const std::function<void(const std::vector<std::vector<size_t>> &,
                             const std::vector<size_t> &)> &window_f) {
  size_t total_rank = in_shape.size();
  size_t spatial_rank = kernel_shape.size();

  std::vector<size_t> out_idx(total_rank, 0);

  std::function<void(size_t)> recurse = [&](size_t dim) {
    if (dim == total_rank) {  // Depth reached

      std::vector<std::vector<size_t>> window_in_idx;
      std::vector<int> kernel_pos(spatial_rank, 0);

      std::function<void(size_t)> kernel_recurse = [&](size_t kdim) {
        if (kdim == spatial_rank) {  // Depth reached
          bool valid = true;
          std::vector<size_t> in_idx(total_rank, 0);
          in_idx[0] = out_idx[0];  // Batch
          in_idx[1] = out_idx[1];  // Channel

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

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE