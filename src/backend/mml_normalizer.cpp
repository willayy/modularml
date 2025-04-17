#include <stddef.h>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include "backend/dataloader/normalizer.hpp"
#include "datastructures/a_tensor.hpp"
#include "datastructures/mml_array.hpp"
#include "datastructures/tensor_factory.hpp"

std::shared_ptr<Tensor<float>> Normalize::normalize(
    const std::shared_ptr<Tensor<float>>& input,
    const std::array<float, 3>& mean, const std::array<float, 3>& std) const {
  auto shape = input->get_shape();

  // Check for exceptions:
  if (shape.size() != 4) {
    throw std::invalid_argument("Input tensor must have 4 dimensions.");
  }
  if (shape[1] != 3) {
    throw std::invalid_argument("Input tensor must have 3 channels (C == 3).");
  }

  size_t N = shape[0];
  size_t C = shape[1];
  size_t H = shape[2];
  size_t W = shape[3];

  auto output = TensorFactory::create_tensor<float>({N, C, H, W});

  // Normalize the input tensor:
  for (size_t n = 0; n < shape[0]; ++n) {
    for (size_t c = 0; c < shape[1]; ++c) {
      for (size_t h = 0; h < shape[2]; ++h) {
        for (size_t w = 0; w < shape[3]; ++w) {
          float v = (*input)[{n, c, h, w}];
          (*output)[{n, c, h, w}] = (v - mean[c]) / std[c];
        }
      }
    }
  }

  return output;
}
