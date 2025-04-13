#include <gtest/gtest.h>

#include <fstream>
#include <modularml>

#include "backend/dataloader/image_loader.hpp"
#include "backend/dataloader/normalizer.hpp"

int getCaffeLabel(const nlohmann::json& jsonData, const std::string& imageKey) {
  if (!jsonData.contains(imageKey)) {
    throw std::runtime_error("Image key not found in JSON");
  }

  const auto& labels = jsonData.at(imageKey);

  for (auto it = labels.begin(); it != labels.end(); ++it) {
    if (it.key().find("CAFFE") != std::string::npos) {
      return it.value();
    }
  }

  throw std::runtime_error("No CAFFE label found for the image");
}

std::string padNumber(int num, int width = 8) {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << num;
    return ss.str();
}

std::pair<size_t, size_t> imageNet(const size_t startingindex, const size_t endingindex) {
  if (endingindex <= startingindex) {
    throw std::invalid_argument("Ending index must be larger than starting index");
  } else if (startingindex > 50000) {
    throw std::invalid_argument("Starting index must be less than 50000");
  } else if (endingindex > 50000) {
    throw std::invalid_argument("Ending index must be less than 50000");
  } else if (startingindex <= 0) {
    throw std::invalid_argument("Starting index must be larger than 0");
  } else if (endingindex < 0) {
    throw std::invalid_argument("Ending index must be larger than 0");
  }

  size_t success = 0;
  size_t failure = 0;
  std::string imagePath = "../tests/data/imagenet/images/";
  std::string labelPath = "../tests/data/imagenet/labels.json";
  std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
  Parser_mml parser;
  Arithmetic_mml<float> arithmetic_instance;

  // Parse and load AlexNet
  std::ifstream file("../alexnet.json");
  nlohmann::json onnx_model;
  file >> onnx_model;
  file.close();
  std::unique_ptr<Model> model_base;
  model_base = parser.parse(onnx_model);
  auto model = dynamic_cast<Model_mml*>(model_base.get());
  std::unordered_map<std::string, GeneralDataTypes> inputs;
  std::unordered_map<std::string, GeneralDataTypes> outputs;

  // loop through images, load them, and run inference
  for (size_t i = startingindex; i < endingindex; ++i) {
    // format the string correctly
    std::string imageFile = "ILSVRC2012_val_" + padNumber(i) + ".JPEG";
    std::string imageFilepath = imagePath + imageFile;
    std::cout << "Processing image: " << imageFile << std::endl;

    // Turn the image into a tensor
    const ImageLoaderConfig config(imageFile);
    auto image_tensor = loader->load(config);

    // Normalize the image (?)

    // Set the input for the model
    inputs["input"] = image_tensor;

    // Run inference
    outputs = model->infer(inputs);

    // Get the output tensor & run arg_max
    auto output_it = outputs.find("output");
    auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
    int result = arithmetic_instance.arg_max(output_tensor);

    // Get the class number from the JSON file
    int expected_result = getCaffeLabel(labelPath, imageFile);

    // Check if it matches the prediction
    // Increase success or failure
    if (result == expected_result) {
      success++;
    } else {
      failure++;
    }
  }
  return {success, failure};
}

TEST(test_imageNet, imageNet) {
  auto result = imageNet(1, 2);
  std::cout << "Success: " << result.first << ", Failure: " << result.second << std::endl;
  EXPECT_EQ(result.first, 1);
  EXPECT_EQ(result.second, 0);
}