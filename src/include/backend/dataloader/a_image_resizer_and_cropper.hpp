#pragma once

#include "backend/dataloader/data_loader_config.hpp"
#include "globals.hpp"

/**
 * @class resize_and_crop
 * @brief Abstract base class for image handling in preparation for machine learning models.
 *
 * This class resizes and crops images to a specified size.
 * Provides a method for resizing data and a virtual destructor.
 *
 * @author Måns Bremer (@Breman402)
 */
class Resize_and_crop {
 public:
  Resize_and_crop() = default;

  /**
   * @brief Load external image data and resize/crop it to fit model input.
   *
   * This is a pure virtual function that must be implemented by derived classes.
   *
   * @return A pointer to resized and cropped image data (allocated with `new`).
   */
  virtual unsigned char* resize_and_crop_image(const DataLoaderConfig& config, int& out_width, int& out_height, int& out_channels) const = 0;

  /**
   * @brief Virtual destructor.
   *
   * Ensures proper cleanup of derived class objects.
   */
  virtual ~Resize_and_crop() = default;
};
