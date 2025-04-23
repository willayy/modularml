#pragma once

#include "dataloader/a_data_loader.hpp"
#include "datastructures/mml_tensor.hpp"

/**
 * @class ImageLoader
 * @brief A class that loads and translates images into tensors.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
class ImageLoader : public DataLoader<float> {
 public:
  
  /**
   * @struct RawImageBuffer
   * @brief Represents a raw image buffer containing image data and its properties.
   *
   * This structure is used to store the raw pixel data of an image along with
   * its dimensions and the number of color channels.
   *
   */
  struct RawImageBuffer {
    std::shared_ptr<unsigned char> data;
    int width;
    int height;
    int channels;
  };

  /**
   * @brief Loads an image according to the configuration and returns a shared
   * pointer to a tensor representation of the original image.
   *
   * @param config The configuration object used to load the image
   * @return A std::shared_ptr to a Tensor containing the loaded data.
   */
  std::shared_ptr<Tensor<float>> load(const DataLoaderConfig &config) const override;

  /**
   * @brief Loads a raw image buffer into a tensor representation.
   * 
   * This function takes a raw image buffer as input and converts it into a 
   * tensor of type `float`. The tensor can then be used for further processing 
   * in machine learning or other computational tasks.
   * 
   * @param raw The raw image buffer to be loaded. It contains the image data 
   *            that needs to be converted into a tensor.
   * @return A shared pointer to a `Tensor<float>` object containing the 
   *         processed image data.
   */
  std::shared_ptr<Tensor<float>> load(const RawImageBuffer &raw) const;
};