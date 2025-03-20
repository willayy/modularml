#include "image_loader.hpp"

template <typename T>
unique_ptr<Tensor<T>> ImageLoader<T>::load(std::string image_path, int width, int height) const {
    int in_width, in_height, channels;

    unsigned char* image_data = stbi_load(image_path, &in_width, &in_height, &channels, 0);

    if (!data) {
        throw invalid_argument("Failed to load image: " + image_path);
    }

    unsigned char* resized_image_data = new unsigned char[width * height * channels];

    if (!stbir_resize_uint8(image_data, in_width, in_height, 0,
                            resized_image_data, width, height, 0, channels)) {
        throw invalid_argument("Failed to resize image");
    }

    // Prepare the output tensor
    array_mml<int> image_tensor_shape({1, channels, width, height});
    array_mml<float> image_data(channels * width * height); // Just 0:s
    shared_ptr<Tensor_mml<float>> output = make_shared<Tensor_mml<float>>(image_tensor_shape, image_data);

    // The data inside image_data is {R, G, B, R, G, B, ...} 
    // So we iterate 3 steps each time and write the R G B for each pixel to the tensor
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * channels;
            
            // This writes each pixel component to the correct slice in the tensor
            for (int c = 0; c < channels; c++) {
                unsigned char pixel_component = image_data[index + c];
                (*output)[{1, c, y, x}] = pixel_component;
            }
        }   
    }
    // Write the image data in the correct order to the tensor data array



}