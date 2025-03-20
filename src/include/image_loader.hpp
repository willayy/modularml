#include "a_data_loader.hpp"
#include "globals.hpp"
#include "stb_image.h"
#include "stb_image_resize.h"

template <typename T>
class ImageLoader : public DataLoader {
   public:
    /**
     * @brief Loads an image.
     *
     * Based on width and height the image is resized.
     *
     * @param path The relative path to the image
     * @param width The desired width of the output
     * @param height The desired width of the output
     * @return A unique_ptr to a Tensor containing the data that was loaded.
     */
    unique_ptr<Tensor<T>> load(std::string image_path, int width, int height) const override;

   private:
};