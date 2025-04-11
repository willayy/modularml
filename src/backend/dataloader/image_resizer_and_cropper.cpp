#include "backend/dataloader/resize_and_cropper.hpp"
#include "stb_image.h"
#include "stb_image_resize2.h"
#include <iostream>


unsigned char* Image_resize_and_cropper::resize_and_crop_image(const DataLoaderConfig& config, int& out_width, int& out_height, int& out_channels) const {
    const ImageLoaderConfig& image_config = dynamic_cast<const ImageLoaderConfig&>(config);

    int width, height, channels;
    unsigned char* input = stbi_load(image_config.image_path.c_str(), &width, &height, &channels, 3);  // force RGB
    if (!input) {
        std::cerr << "Failed to load image: " << image_config.image_path << "\n";
        return nullptr;
    }

    // Resize: shortest side to 256, preserve aspect ratio
    const int resize_short = 256;
    int new_width, new_height;
    if (width < height) {
        new_width = resize_short;
        new_height = static_cast<int>(height * (resize_short / static_cast<float>(width)));
    } else {
        new_height = resize_short;
        new_width = static_cast<int>(width * (resize_short / static_cast<float>(height)));
    }

    // Allocate resized buffer
    std::vector<unsigned char> resized(new_width * new_height * 3);
    int input_stride = width * 3;
    int output_stride = new_width * 3;

    // Resize using stb_image_resize2
    if (!stbir_resize_uint8_linear(
            input, width, height, input_stride,
            resized.data(), new_width, new_height, output_stride,
            STBIR_RGB)) {
        std::cerr << "stbir_resize_uint8_linear failed\n";
        stbi_image_free(input);
        return nullptr;
    }

    stbi_image_free(input);

    // Center crop 224x224
    const int crop_size = 224;
    if (new_width < crop_size || new_height < crop_size) {
        std::cerr << "Resized image smaller than crop size\n";
        return nullptr;
    }

    int x_offset = (new_width - crop_size) / 2;
    int y_offset = (new_height - crop_size) / 2;

    unsigned char* cropped = new unsigned char[crop_size * crop_size * 3];
    for (int y = 0; y < crop_size; ++y) {
        for (int x = 0; x < crop_size; ++x) {
            for (int c = 0; c < 3; ++c) {
                cropped[(y * crop_size + x) * 3 + c] =
                    resized[((y + y_offset) * new_width + (x + x_offset)) * 3 + c];
            }
        }
    }

    out_width = crop_size;
    out_height = crop_size;
    out_channels = 3;
    return cropped;
}
