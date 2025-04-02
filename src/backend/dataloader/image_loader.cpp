#include "backend/dataloader/image_loader.hpp"

// Is needed to create the implementation
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


std::shared_ptr<Tensor<float>> ImageLoader::load(const DataLoaderConfig& config) const {
    const ImageLoaderConfig& image_config = dynamic_cast<const ImageLoaderConfig&>(config);

    int width, height, channels;

    unsigned char* image_data = stbi_load(image_config.image_path.c_str(), &width, &height, &channels, 0);

    if (!image_data) {
        throw std::invalid_argument("Failed to load image: " + image_config.image_path);
    }

    // Trust
    int output_channels = channels;

    int data_size = width * height * channels;
    float* float_image_data = new float[data_size];
    for (int i = 0; i < data_size; i++) {
        float_image_data[i] = static_cast<float>(image_data[i]) / 255.0; // Here we normalize the RGB value to between 0.0 - 1.0.
    }

    // Prepare output tensor
    
    if (!image_config.include_alpha_channel && channels == 4) {
        output_channels = 3;
    }

    array_mml<unsigned long int> image_tensor_shape({1, static_cast<unsigned long int>(output_channels), static_cast<unsigned long int>(width), static_cast<unsigned long int>(height)});
    array_mml<float> output_data(data_size);  // Fills with 0:s
    std::shared_ptr<Tensor_mml<float>> output = std::make_shared<Tensor_mml<float>>(image_tensor_shape, output_data);

    // The data inside output_data is {R, G, B, R, G, B, ...} 
    // So we iterate 3 steps each time and write the R G B for each pixel to the tensor
    for (unsigned long int y = 0; y < height; y++) {
        for (unsigned long int x = 0; x < width; x++) {
            int index = (y * width + x) * channels;
            
            // This writes each pixel component to the correct slice in the tensor
            for (unsigned long int c = 0; c < channels && c < 3; c++) { //XD
                float pixel_component = float_image_data[index + c];
                
                // Write the pixel component value, we assume that only a single image is loaded at a time currently
                (*output)[{0, c, y, x}] = pixel_component;
            }
        }   
    }
    return output; // Return the shared pointer
}


