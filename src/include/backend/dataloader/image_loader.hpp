#pragma once

#include "backend/dataloader/a_data_loader.hpp"
#include "datastructures/tensor.hpp"

/**
 * @class ImageLoader
 * @brief A class that loads and translates images into tensors.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
class ImageLoader : public DataLoader<float> {
public:
  /**
   * @brief Loads an image according to the configuration and returns a shared
   * pointer to a tensor representation of the original image.
   *
   * @param config The configuration object used to load the image
   * @return A std::shared_ptr to a Tensor containing the loaded data.
   */
  std::shared_ptr<Tensor<float>> load(const DataLoaderConfig &config) const;
};