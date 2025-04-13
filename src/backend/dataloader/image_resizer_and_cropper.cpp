#include <iostream>

#include "backend/dataloader/resize_and_cropper.hpp"
#include "stb_image.h"
#include "stb_image_resize2.h"

namespace {

// Custom deleter for shared_ptr to correctly free stbi-loaded memory
auto stbi_deleter = [](unsigned char* data) {
  stbi_image_free(data);
};

}  // namespace

// Resize and return shared_ptr-managed image buffer
std::shared_ptr<unsigned char> Image_resize_and_cropper::resize(const DataLoaderConfig& config, int& out_width, int& out_height, int& out_channels) const {
  const ImageLoaderConfig& image_config = dynamic_cast<const ImageLoaderConfig&>(config);

  int width, height, channels;
  unsigned char* input = stbi_load(image_config.image_path.c_str(), &width, &height, &channels, 3);  // force RGB
  if (!input) {
    std::cerr << "Failed to load image: " << image_config.image_path << "\n";
    return nullptr;
  }

  const int resize_short = 256;
  int new_width, new_height;
  if (width < height) {
    new_width = resize_short;
    new_height = static_cast<int>(height * (resize_short / static_cast<float>(width)));
  } else {
    new_height = resize_short;
    new_width = static_cast<int>(width * (resize_short / static_cast<float>(height)));
  }

  std::vector<unsigned char> resized(new_width * new_height * 3);
  int input_stride = width * 3;
  int output_stride = new_width * 3;

  if (!stbir_resize_uint8_linear(
          input, width, height, input_stride,
          resized.data(), new_width, new_height, output_stride,
          STBIR_RGB)) {
    std::cerr << "stbir_resize_uint8_linear failed\n";
    stbi_image_free(input);
    return nullptr;
  }

  // Free original image memory now that we're done with it
  stbi_image_free(input);

  // Allocate shared_ptr with new[] and custom deleter
  size_t num_bytes = resized.size();
  std::shared_ptr<unsigned char> output(new unsigned char[num_bytes], std::default_delete<unsigned char[]>());
  std::copy(resized.begin(), resized.end(), output.get());

  out_width = new_width;
  out_height = new_height;
  out_channels = 3;
  return output;
}

std::shared_ptr<unsigned char> Image_resize_and_cropper::crop(
    const std::shared_ptr<unsigned char>& resized_data,
    int width, int height, int channels, int crop_size) const {
  if (width < crop_size || height < crop_size) {
    std::cerr << "Image is smaller than the crop size\n";
    return nullptr;
  }

  int x_offset = (width - crop_size) / 2;
  int y_offset = (height - crop_size) / 2;

  size_t num_bytes = crop_size * crop_size * channels;
  std::shared_ptr<unsigned char> cropped(new unsigned char[num_bytes], std::default_delete<unsigned char[]>());

  unsigned char* cropped_ptr = cropped.get();
  const unsigned char* resized_ptr = resized_data.get();

  for (int y = 0; y < crop_size; ++y) {
    for (int x = 0; x < crop_size; ++x) {
      for (int c = 0; c < channels; ++c) {
        cropped_ptr[(y * crop_size + x) * channels + c] =
            resized_ptr[((y + y_offset) * width + (x + x_offset)) * channels + c];
      }
    }
  }

  return cropped;
}
