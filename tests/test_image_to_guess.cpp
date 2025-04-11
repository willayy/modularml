#include <gtest/gtest.h>

#include <fstream>
#include <modularml>

#include "backend/dataloader/image_loader.hpp"
#include "backend/dataloader/normalizer.hpp"
#include "backend/dataloader/resize_and_cropper.hpp"
#include "stb_image_write.h"

std::string getClassName(const std::string& filename, const std::string& id) {
  std::ifstream file(filename);
  nlohmann::json j;
  file >> j;
  return j[id][1];  // Second element is the human-readable class name
}

TEST(test_alexnet_image_to_guess, image_to_guess) {
  // Image to be processed:
  string input_path = "tests/test_image_to_guess.png";
  const ImageLoaderConfig config("../tests/data/alexnet/alexnet_pictures/foxhound.png");

  // Resize and crop the image for Alexnet:
  Image_resize_and_cropper resizer_and_cropper;
  int out_width, out_height, out_channels;
  unsigned char* resized_cropped_image = resizer_and_cropper.resize_and_crop_image(config, out_width, out_height, out_channels);
  // Check the dimensions of the resized and cropped image
  EXPECT_EQ(out_width, 224);
  EXPECT_EQ(out_height, 224);
  EXPECT_EQ(out_channels, 3);

  // Save the resized and cropped image to a temporary file
  std::string temp_image_path = "../tests/data/alexnet/alexnet_pictures/temp_test_image.png";
  stbi_write_png(temp_image_path.c_str(), out_width, out_height, out_channels, resized_cropped_image, out_width * out_channels);
  const ImageLoaderConfig resized_config(temp_image_path);

  // Load the temporary resized image using ImageLoader
  // in order to create a tensor from it:
  shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
  auto image_tensor = loader->load(resized_config);

  // Delete the temporary image file
  std::remove(temp_image_path.c_str());

  // Normalize the image tensor:
  Normalize normalizer;
  auto normalized_tensor = normalizer.normalize(image_tensor, {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});

  // Parse Alexnet:
  std::ifstream file("../alexnet.json");
  ASSERT_TRUE(file.is_open()) << "Failed to open alexnet.json file";
  json onnx_model;
  file >> onnx_model;
  file.close();
  Parser_mml parser;
  std::unique_ptr<Model> model_base;
  ASSERT_NO_THROW({ model_base = parser.parse(onnx_model); }) << "Parser failed to parse the JSON file";

  auto model = dynamic_cast<Model_mml*>(model_base.get());

  // Set the loaded image as an input to the model:
  std::unordered_map<string, GeneralDataTypes> inputs;
  inputs["input"] = normalized_tensor;

  // Perform inference:
  std::unordered_map<string, GeneralDataTypes> outputs;
  outputs = model->infer(inputs);

  // Handle the output:
  auto output_it = outputs.find("output");
  auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);

  // Take the result from the output tensor using argmax:
  Arithmetic_mml<float> arithmetic_instance;
  int result = arithmetic_instance.arg_max(output_tensor);

  std::string class_name = getClassName("../tests/data/alexnet/alexnet_ImageNet_labels.json", std::to_string(result));
  std::cout << "Predicted class: " << class_name << std::endl;
  ASSERT_TRUE(class_name == "English_foxhound") << "Predicted class does not match expected class";
}