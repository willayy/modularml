#pragma once

#include "datastructures/a_tensor.hpp"

/**
 * @class Normalizer
 * @brief Abstract base class for normalizing a tensor.
 *
 * This class defines the interface for normalizing a tensor, providing a method
 * for normalization
 *
 * @author Måns Bremer (@Breman402)
 */
template <typename InputT, typename OutputT>
class Normalizer {
 public:
  Normalizer() = default;

  /**
   * @brief Normalize a tensor and return it.
   *
   * This is a pure virtual function that must be implemented by derived
   * classes.
   *
   * @param tensor The tensor to be normalized.
   * @param mean An array (len. 3) of mean values for each channel.
   * @param std An array (len. 3) of standard deviation values for each channel.
   * @return A unique_ptr to a Tensor containing the data.
   */
  virtual std::shared_ptr<Tensor<OutputT>> normalize(
      const std::shared_ptr<Tensor<InputT>>& input,
      const std::array<float, 3>& mean,
      const std::array<float, 3>& std) const = 0;

  /**
   * @brief Virtual destructor for the Normalizer class.
   *
   * Ensures proper cleanup of derived class objects.
   */
  virtual ~Normalizer() = default;
};