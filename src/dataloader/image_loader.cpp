#include "../include/dataloader/image_loader.hpp"

#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <memory>
// IWYU pragma: no_include <__ostream/basic_ostream.h>
#include <ostream>  // IWYU pragma: keep
#include <stdexcept>
#include <string>

#include "../include/dataloader/data_loader_config.hpp"
#include "datastructures/a_tensor.hpp"
#include "datastructures/mml_array.hpp"
#include "datastructures/mml_tensor.hpp"
#include "datastructures/tensor_factory.hpp"
#include "stb_image.h"

std::shared_ptr<Tensor<float>> ImageLoader::load(
    const DataLoaderConfig &config) const {
  const ImageLoaderConfig &image_config =
      dynamic_cast<const ImageLoaderConfig &>(config);

  int width;
  int height;
  int channels;

  unsigned char *image_data =
      stbi_load(image_config.image_path.c_str(), &width, &height, &channels, 0);

  if (!image_data) {
    throw std::invalid_argument("Failed to load image: " +
                                image_config.image_path);
  }
  
  // Trust
  int output_channels = channels;

  int data_size = width * height * channels;
  std::vector<float> float_image_data = std::vector<float>(data_size);
  for (int i = 0; i < data_size; i++) {
    float_image_data[i] =
        static_cast<float>(image_data[i]) /
        255.0f;  // Here we normalize the RGB value to between 0.0 - 1.0.
  }

  // Prepare output tensor

  if (!image_config.include_alpha_channel && channels == 4) {
    output_channels = 3;
  }

  array_mml<unsigned long int> image_tensor_shape(
      {1, static_cast<unsigned long int>(output_channels),
       static_cast<unsigned long int>(height),
       static_cast<unsigned long int>(width)});
  array_mml<float> output_data(data_size);  // Fills with 0:s
  std::shared_ptr<Tensor<float>> output =
      TensorFactory::create_tensor<float>(image_tensor_shape, output_data);

  // The data inside output_data is {R, G, B, R, G, B, ...}
  // So we iterate 3 steps each time and write the R G B for each pixel to the
  // tensor
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int index = (y * width + x) * channels;

      // This writes each pixel component to the correct slice in the tensor
      for (int c = 0; c < channels && c < 3; c++) {  // XD
        float pixel_component = float_image_data[index + c];

        // Write the pixel component value, we assume that only a single image
        // is loaded at a time currently
        (*output)[{0, static_cast<unsigned long int>(c),
                   static_cast<unsigned long int>(y),
                   static_cast<unsigned long int>(x)}] = pixel_component;
      }
    }
  }

  return output;  // Return the shared pointer
}

std::shared_ptr<Tensor<float>> ImageLoader::load(
    const RawImageBuffer &raw) const {
  if (!raw.data) {
    throw std::invalid_argument("ImageLoader: raw image data is null");
  }

  int data_size = raw.width * raw.height * raw.channels;
  std::vector<float> float_image_data = std::vector<float>(data_size);

  for (int i = 0; i < data_size; ++i) {
    float_image_data[i] = static_cast<float>(raw.data.get()[i]) / 255.0f;
  }

  int output_channels = raw.channels == 4 ? 3 : raw.channels;
  array_mml<unsigned long int> image_tensor_shape(
      {1, static_cast<unsigned long int>(output_channels),
       static_cast<unsigned long int>(raw.height),
       static_cast<unsigned long int>(raw.width)});
  array_mml<float> output_data(data_size);
  std::shared_ptr<Tensor_mml<float>> output =
      std::make_shared<Tensor_mml<float>>(image_tensor_shape, output_data);

  for (int y = 0; y < raw.height; ++y) {
    for (int x = 0; x < raw.width; ++x) {
      int index = (y * raw.width + x) * raw.channels;
      for (int c = 0; c < raw.channels && c < 3; ++c) {
        float pixel_component = float_image_data[index + c];
        (*output)[{0, static_cast<unsigned long int>(c),
                   static_cast<unsigned long int>(y),
                   static_cast<unsigned long int>(x)}] = pixel_component;
      }
    }
  }

  return output;
}
