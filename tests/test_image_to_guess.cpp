#include <gtest/gtest.h>

#include <fstream>
#include <modularml>

#include "backend/dataloader/image_loader.hpp"
#include "backend/dataloader/normalizer.hpp"
#include "backend/dataloader/resize_and_cropper.hpp"
#include "stb_image_write.h"

std::string get_class_name(const std::string& filename, const std::string& id) {
  std::ifstream file(filename);
  nlohmann::json j;
  file >> j;
  return j[id][1];
}

TEST(test_alexnet_image_to_guess, image_to_guess) {
  std::string input_path = "tests/test_image_to_guess.png";
  const ImageLoaderConfig config("../tests/data/alexnet/alexnet_pictures/foxhound.png");

  Image_resize_and_cropper resizer_and_cropper;
  int out_width, out_height, out_channels;
  std::shared_ptr<unsigned char> resized_image = resizer_and_cropper.resize(config, out_width, out_height, out_channels);

  const int crop_size = 224;
  std::shared_ptr<unsigned char> resized_cropped_image =
      resizer_and_cropper.crop(resized_image, out_width, out_height, out_channels, crop_size);

  out_width = crop_size;
  out_height = crop_size;

  EXPECT_EQ(out_width, 224);
  EXPECT_EQ(out_height, 224);
  EXPECT_EQ(out_channels, 3);

  std::string temp_image_path = "../tests/data/alexnet/alexnet_pictures/temp_test_image.png";
  stbi_write_png(temp_image_path.c_str(), out_width, out_height, out_channels, resized_cropped_image.get(), out_width * out_channels);
  const ImageLoaderConfig resized_config(temp_image_path);

  std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
  auto image_tensor = loader->load(resized_config);

  std::remove(temp_image_path.c_str());

  Normalize normalizer;
  auto normalized_tensor = normalizer.normalize(image_tensor, {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});

  std::ifstream file("../alexnet.json");
  if (!file.is_open()) {
    GTEST_SKIP() << "Skipping test as alexnet.json file is not found";
  }
  nlohmann::json onnx_model;
  file >> onnx_model;
  file.close();

  Parser_mml parser;
  std::unique_ptr<Model> model_base;
  ASSERT_NO_THROW({ model_base = parser.parse(onnx_model); }) << "Parser failed to parse the JSON file";
  auto model = dynamic_cast<Model_mml*>(model_base.get());

  std::unordered_map<std::string, GeneralDataTypes> inputs;
  inputs["input"] = normalized_tensor;

  std::unordered_map<std::string, GeneralDataTypes> outputs;
  outputs = model->infer(inputs);

  auto output_it = outputs.find("output");
  auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);

  Arithmetic_mml<float> arithmetic_instance;
  int result = arithmetic_instance.arg_max(output_tensor);

  std::string class_name = get_class_name("../tests/data/alexnet/alexnet_ImageNet_labels.json", std::to_string(result));
  std::cout << "Predicted class: " << class_name << std::endl;
  ASSERT_TRUE(class_name == "English_foxhound") << "Predicted class does not match expected class";
}

TEST(test_alexnet_image_to_guess, image_to_guess2) {
  std::string input_path = "tests/test_image_to_guess.png";
  const ImageLoaderConfig config("../tests/data/alexnet/alexnet_pictures/American_egret.png");

  Image_resize_and_cropper resizer_and_cropper;
  int out_width, out_height, out_channels;
  std::shared_ptr<unsigned char> resized_image = resizer_and_cropper.resize(config, out_width, out_height, out_channels);

  const int crop_size = 224;
  std::shared_ptr<unsigned char> resized_cropped_image =
      resizer_and_cropper.crop(resized_image, out_width, out_height, out_channels, crop_size);

  out_width = crop_size;
  out_height = crop_size;

  EXPECT_EQ(out_width, 224);
  EXPECT_EQ(out_height, 224);
  EXPECT_EQ(out_channels, 3);

  std::string temp_image_path = "../tests/data/alexnet/alexnet_pictures/temp_test_image.png";
  stbi_write_png(temp_image_path.c_str(), out_width, out_height, out_channels, resized_cropped_image.get(), out_width * out_channels);
  const ImageLoaderConfig resized_config(temp_image_path);

  std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
  auto image_tensor = loader->load(resized_config);

  std::remove(temp_image_path.c_str());

  Normalize normalizer;
  auto normalized_tensor = normalizer.normalize(image_tensor, {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});

  std::ifstream file("../alexnet.json");
  if (!file.is_open()) {
    GTEST_SKIP() << "Skipping test as alexnet.json file is not found";
  }
  nlohmann::json onnx_model;
  file >> onnx_model;
  file.close();

  Parser_mml parser;
  std::unique_ptr<Model> model_base;
  ASSERT_NO_THROW({ model_base = parser.parse(onnx_model); }) << "Parser failed to parse the JSON file";
  auto model = dynamic_cast<Model_mml*>(model_base.get());

  std::unordered_map<std::string, GeneralDataTypes> inputs;
  inputs["input"] = normalized_tensor;

  std::unordered_map<std::string, GeneralDataTypes> outputs;
  outputs = model->infer(inputs);

  auto output_it = outputs.find("output");
  auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);

  Arithmetic_mml<float> arithmetic_instance;
  int result = arithmetic_instance.arg_max(output_tensor);

  std::string class_name = get_class_name("../tests/data/alexnet/alexnet_ImageNet_labels.json", std::to_string(result));
  std::cout << "Predicted class: " << class_name << std::endl;
  ASSERT_TRUE(class_name == "American_egret") << "Predicted class does not match expected class";
}
