#pragma once 

#include "backend/dataloader/a_data_loader.hpp"
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
     * @brief Loads an image.
     *
     * Based on width and height the image is resized.
     *
     * @param config The relative path to the image
     * @return A unique_ptr to a Tensor containing the loaded data.
     */
    std::shared_ptr<Tensor<float>> load(const DataLoaderConfig& config) const;
};