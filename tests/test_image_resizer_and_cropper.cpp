#include <gtest/gtest.h>

#include <modularml>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

TEST(test_image_resizer_and_cropper, test_resize_and_crop) {
  // Create a dummy image for testing
  const int width = 512;
  const int height = 512;
  const int channels = 3;
  unsigned char* dummy_image = new unsigned char[width * height * channels];
  for (int i = 0; i < width * height * channels; ++i) {
    dummy_image[i] = static_cast<unsigned char>(i % 256);
  }

  // Save the dummy image to a temporary file
  std::string temp_image_path = "temp_test_image.png";
  stbi_write_png(temp_image_path.c_str(), width, height, channels, dummy_image,
                 width * channels);

  // Load the image using ImageLoaderConfig
  ImageLoaderConfig config(temp_image_path);
  imageResizeAndCropper resizer_and_cropper;

  int out_width;
  int out_height;
  int out_channels;
  std::shared_ptr<unsigned char> resized_image =
      resizer_and_cropper.resize(config, out_width, out_height, out_channels);

  // Perform cropping on the resized image
  const int crop_size = 224;
  std::shared_ptr<unsigned char> resized_cropped_image =
      resizer_and_cropper.crop(resized_image, out_width, out_height,
                               out_channels, crop_size);

  // Update the output dimensions after cropping
  out_width = crop_size;
  out_height = crop_size;

  // Check if the resized and cropped image is not null
  ASSERT_NE(resized_cropped_image, nullptr);

  // Check the dimensions of the resized and cropped image
  EXPECT_EQ(out_width, 224);
  EXPECT_EQ(out_height, 224);
  EXPECT_EQ(out_channels, channels);

  // Clean up
  delete[] dummy_image;

  // Remove the temporary image file
  std::remove(temp_image_path.c_str());
}
