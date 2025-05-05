#pragma once

#include "datastructures/mml_tensor.hpp"
#include "normalizer/a_normalizer.hpp"

/**
 * @class Normalizer_mml
 * @brief Concrete implementation of tensor normalization for float tensors.
 *
 * This class implements the Normalizer interface specifically for
 * floating-point tensors. It provides channel-wise normalization by applying
 * mean subtraction and standard deviation division, which is a common
 * preprocessing step in machine learning pipelines, particularly for image
 * data.
 *
 * @author Måns Bremer (@Breman402)
 */
class Normalizer_mml : public Normalizer<float, float> {
 public:
  /**
   * @brief Normalizes a float tensor using specified mean and standard
   * deviation values.
   *
   * This method implements the normalization algorithm for floating-point
   * tensors. For each element in the input tensor, it applies the formula:
   * normalized_value = (original_value - mean[channel]) / std[channel]
   *
   * For 3-channel image data (like RGB), the normalization is applied
   * channel-wise, using the corresponding mean and standard deviation values
   * for each channel.
   *
   * @param input The float tensor to be normalized
   * @param mean An array of mean values for each channel (typically RGB
   * channels for images)
   * @param std An array of standard deviation values for each channel
   * @return A shared pointer to the normalized float tensor
   */
  std::shared_ptr<Tensor<float>> normalize(
      const std::shared_ptr<Tensor<float>>& input,
      const std::array<float, 3>& mean,
      const std::array<float, 3>& std) const override;
};