#pragma once

#include "datastructures/a_tensor.hpp"

/**
 * @class Normalizer
 * @brief Abstract base class for tensor normalization operations.
 *
 * This template class defines the interface for normalizing tensor data,
 * which is a common preprocessing step in machine learning pipelines.
 * Normalization typically involves transforming raw input values to a
 * standardized range, often using mean subtraction and division by
 * standard deviation.
 *
 * @tparam InputT The data type of the input tensor elements
 * @tparam OutputT The data type of the output tensor elements after
 * normalization
 * @author Måns Bremer (@Breman402)
 */
template <typename InputT, typename OutputT>
class Normalizer {
 public:
  /**
   * @brief Default constructor for Normalizer.
   */
  Normalizer() = default;

  /**
   * @brief Normalizes a tensor using specified mean and standard deviation
   * values.
   *
   * This pure virtual function must be implemented by derived classes to define
   * specific normalization algorithms. A typical normalization formula is:
   * normalized_value = (original_value - mean) / std
   *
   * @param input The tensor to be normalized
   * @param mean An array of mean values for each channel (typically RGB
   * channels for images)
   * @param std An array of standard deviation values for each channel
   * @return A shared pointer to a normalized tensor with elements of type
   * OutputT
   */
  virtual std::shared_ptr<Tensor<OutputT>> normalize(
      const std::shared_ptr<Tensor<InputT>>& input,
      const std::array<float, 3>& mean,
      const std::array<float, 3>& std) const = 0;

  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes.
   *
   * This allows safe polymorphic destruction of normalizer objects when
   * accessed through a pointer to this base class.
   */
  virtual ~Normalizer() = default;
};