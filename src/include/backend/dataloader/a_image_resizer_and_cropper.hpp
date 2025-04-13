#pragma once

#include <memory>

#include "backend/dataloader/data_loader_config.hpp"

/**
 * @class resizeAndCrop
 * @brief Abstract base class for image preprocessing in machine learning pipelines.
 *
 * This class defines an interface for resizing and cropping image data.
 * It uses shared pointers for automatic and safe memory management of image buffers.
 * Derived classes must implement both methods.
 *
 * @author Måns Bremer (@Breman402)
 */
class resizeAndCrop {
 public:
  resizeAndCrop() = default;

  /**
   * @brief Resizes an image based on the provided configuration.
   *
   * This is a pure virtual function that must be implemented by subclasses.
   * The image is typically resized to prepare it for model input, preserving
   * aspect ratio or fitting a specific dimension.
   *
   * @param config Configuration object specifying how the image should be loaded.
   * @param out_width Output parameter for the resized image width.
   * @param out_height Output parameter for the resized image height.
   * @param out_channels Output parameter for the number of image channels (e.g., 3 for RGB).
   * @return A shared pointer to the resized image data.
   */
  virtual std::shared_ptr<unsigned char> resize(const DataLoaderConfig& config, int& out_width, int& out_height, int& out_channels) const = 0;

  /**
   * @brief Crops a resized image to a centered square region.
   *
   * This is a pure virtual function that must be implemented by subclasses.
   * It performs a center crop of the image buffer to match the required input
   * size of a model.
   *
   * @param resized_data Shared pointer to resized image data.
   * @param width Width of the resized image.
   * @param height Height of the resized image.
   * @param channels Number of channels in the image (e.g., 3 for RGB).
   * @param crop_size Desired size of the square crop.
   * @return A shared pointer to the cropped image data.
   */
  virtual std::shared_ptr<unsigned char> crop(const std::shared_ptr<unsigned char>& resized_data, int width, int height, int channels, int crop_size) const = 0;

  /**
   * @brief Virtual destructor.
   *
   * Ensures proper cleanup of derived class objects when accessed through a base pointer.
   */
  virtual ~resizeAndCrop() = default;
};
