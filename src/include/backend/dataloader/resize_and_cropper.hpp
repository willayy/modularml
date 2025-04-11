#pragma once 

#include "backend/dataloader/a_image_resizer_and_cropper.hpp"

/**
 * @class Resize_and_cropper
 * @brief A class that resizes and crops an image according to the given inputs.
 * 
 * @author Måns Bremer (@Breman402)
 */
class Image_resize_and_cropper : public Resize_and_crop {
   public:
    /**
     * @brief Loads an image according to the configuration and returns a pointer to resized and cropped image data.
     *
     * @param config The configuration object used to load the image
     * @param out_width The width of the resized and cropped image
     * @param out_height The height of the resized and cropped image
     * @param out_channels The number of channels in the resized and cropped image (e.g., 3 for RGB)
     * @return A pointer to resized and cropped image data.
     */
    unsigned char* resize_and_crop_image(const DataLoaderConfig& config, int& out_width, int& out_height, int& out_channels) const;
};