#include "backend/dataloader/normalizer.hpp"

std::shared_ptr<Tensor<float>> Normalize::normalize(
    const std::shared_ptr<Tensor<float>>& input,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& std) const {
  auto shape = input->get_shape();

  // Check for excpetions:
  if (shape.size() != 4) {
    throw std::invalid_argument("Input tensor must have 4 dimensions.");
  }
  if (shape[1] != 3) {
    throw std::invalid_argument("Input tensor must have 3 channels (C == 3).");
  }


  uli N = shape[0];
  uli C = shape[1];
  uli H = shape[2];
  uli W = shape[3];

  auto output = tensor_mml_p<float>({N, C, H, W});

  // Normalize the input tensor:
  for (uli n = 0; n < shape[0]; ++n) {
    for (uli c = 0; c < shape[1]; ++c) {
      for (uli h = 0; h < shape[2]; ++h) {
        for (uli w = 0; w < shape[3]; ++w) {
          float v = (*input)[{n, c, h, w}];
          (*output)[{n, c, h, w}] = (v - mean[c]) / std[c];
        }
      }
    }
  }

  return output;
}
