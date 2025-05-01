#pragma once

#include "backend/dataloader/a_normalizer.hpp"
#include "datastructures/tensor.hpp"

/**
 * @class Normalizer
 * @brief A class that normalizes a tensor and returns it.
 *
 * @author Måns Bremer (@Breman402)
 */
class Normalize : public Normalizer<float, float> {
 public:
  /**
   * @brief normalizes a tensor according to the inputs and returns a shared
   * pointer to a tensor.
   *
   * @param tensor The tensor to be normalized.
   * @param mean An array (len. 3) of mean values for each channel.
   * @param std An array (len. 3) of standard deviation values for each channel.
   * @return A shared_ptr to a Tensor containing the normalized data.
   */
  std::shared_ptr<Tensor<float>> normalize(
      const std::shared_ptr<Tensor<float>>& input,
      const std::array<float, 3>& mean, const std::array<float, 3>& std) const;
};