#include <gtest/gtest.h>

#include <fstream>
#include <modularml>

#include "backend/dataloader/image_loader.hpp"
#include "backend/dataloader/normalizer.hpp"

std::string getClassName(const std::string& filename, const std::string& id) {
  std::ifstream file(filename);
  nlohmann::json j;
  file >> j;
  return j[id][1];  // Second element is the human-readable class name
}

TEST(test_alexnet_image_to_guess, image_to_guess) {
  // Convert an input PNG to a tensor:
  string input_path = "tests/test_image_to_guess.png";
  const ImageLoaderConfig config("../tests/data/alexnet/alexnet_pictures/beagle.png");
  shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
  auto image_tensor = loader->load(config);
  std::cout << image_tensor << std::endl;

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