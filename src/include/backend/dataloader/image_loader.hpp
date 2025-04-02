#pragma once 

#include "backend/dataloader/a_data_loader.hpp"
#include "datastructures/mml_tensor.hpp"
#include "globals.hpp"

/**
 * @class ImageLoader
 * @brief A class that loads and translates images into tensors.
 * 
 * @author Tim Carlsson (timca@chalmers.se)
 */

class ImageLoader : public DataLoader<float> {
   public:
    /**
     * @brief Loads an image.
     *
     * Based on width and height the image is resized.
     *
     * @param path The relative path to the image
     * @return A unique_ptr to a Tensor containing the loaded data.
     */
    unique_ptr<Tensor<float>> load(const ImageLoaderConfig& config) const;

   private:
};