#include "image_loader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

template <typename T>
std::unique_ptr<Tensor<T>> ImageLoader<T>::load(std::string image_path, int width, int height) const {
    int in_width, in_height, channels;

    unsigned char* image_data = stbi_load(image_path.c_str(), &in_width, &in_height, &channels, 0);

    if (!image_data) {
        throw std::invalid_argument("Failed to load image: " + image_path);
    }

    // Convert the unsigned char image data to float data for stbir
    float* float_image_data = new float[in_width * in_height * channels];
    for (int i = 0; i < in_width * in_height * channels; ++i) {
        float_image_data[i] = static_cast<float>(image_data[i]) / 255.0f;  // Normalize to [0, 1]
    }

    // Prepare the output tensor
    array_mml<int> image_tensor_shape({1, channels, width, height});
    array_mml<float> output_data(channels * width * height);  // Initialize a float array
    std::shared_ptr<Tensor_mml<float>> output = std::make_shared<Tensor_mml<float>>(image_tensor_shape, output_data);

    // The data inside output_data is {R, G, B, R, G, B, ...} 
    // So we iterate 3 steps each time and write the R G B for each pixel to the tensor
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * channels;
            
            // This writes each pixel component to the correct slice in the tensor
            for (int c = 0; c < channels; c++) {
                float pixel_component = float_image_data[index + c];
                
                // Write the pixel component value
                (*output)[{1, c, y, x}] = pixel_component;
            }
        }   
    }

    // Return the Tensor as a unique_ptr
    return std::make_unique<Tensor_mml<T>>(*output);
}

template class ImageLoader<float>;
