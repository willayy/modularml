#pragma once

#include "backend/dataloader/a_image_resizer_and_cropper.hpp"

/**
 * @class imageResizeAndCropper
 * @brief A class that resizes and crops an image using shared ownership of image data.
 *
 * This class performs resizing and center-cropping on image data according to
 * configuration parameters. It returns image buffers using shared pointers, ensuring
 * automatic memory management and safe ownership transfer.
 *
 * @author Måns Bremer (@Breman402)
 */
class imageResizeAndCropper : public resizeAndCrop {
 public:
  /**
   * @brief Loads and resizes an image according to the configuration.
   *
   * The image is resized so that its shortest side is 256 pixels while
   * preserving the aspect ratio. The result is returned as a shared pointer,
   * ensuring automatic memory management.
   *
   * @param config The configuration object specifying image loading parameters.
   * @param out_width Output parameter for the width of the resized image.
   * @param out_height Output parameter for the height of the resized image.
   * @param out_channels Output parameter for the number of image channels (e.g., 3 for RGB).
   * @return A shared pointer to the resized image data.
   */
  std::shared_ptr<unsigned char> resize(const DataLoaderConfig& config, int& out_width, int& out_height, int& out_channels) const override;

  /**
   * @brief Crops the resized image data to a centered square of specified size.
   *
   * This function extracts a center-cropped region from the resized image data.
   * It returns the result as a shared pointer for automatic memory management.
   *
   * @param resized_data Shared pointer to the resized image data.
   * @param width The width of the resized image.
   * @param height The height of the resized image.
   * @param channels The number of color channels in the image (e.g., 3 for RGB).
   * @param crop_size The size of the square crop (width and height).
   * @return A shared pointer to the cropped image data.
   */
  std::shared_ptr<unsigned char> crop(const std::shared_ptr<unsigned char>& resized_data, int width, int height, int channels, int crop_size) const override;
};
