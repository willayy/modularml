#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <modularml>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

std::unordered_map<std::string, int> load_labels(const std::string& json_file,
                                                 const std::string& label_key) {
  std::unordered_map<std::string, int> labels_map;
  std::ifstream file(json_file);
  if (file.is_open()) {
    json j;
    file >> j;

    for (auto& el : j.items()) {
      std::string image_name = el.key();
      int label = el.value()[label_key];
      labels_map[image_name] = label;
    }
  } else {
    std::cerr << "Could not open JSON file: " << json_file << std::endl;
  }
  return labels_map;
}

std::string get_class_name(const std::string& filename, const std::string& id) {
  std::ifstream file(filename);
  nlohmann::json j;
  file >> j;
  return j[id][1];
}

std::vector<int> run_alexnet_inference(const std::unique_ptr<Model>& model,
                                       std::string image_path) {
  Profiler::begin_timing("Performing Inference");

  std::unordered_map<std::string, GeneralDataTypes> inputs;
  std::unordered_map<std::string, GeneralDataTypes> outputs;

  try {
    const ImageLoaderConfig config(image_path);

    imageResizeAndCropper resizer_and_cropper;
    int out_width, out_height, out_channels;

    ImageLoader loader;
    std::shared_ptr<unsigned char> resized_image =
        resizer_and_cropper.resize(config, out_width, out_height, out_channels);
    const int crop_size = 224;
    std::shared_ptr<unsigned char> resized_cropped_image =
        resizer_and_cropper.crop(resized_image, out_width, out_height,
                                 out_channels, crop_size);

    out_width = crop_size;
    out_height = crop_size;

    std::string temp_image_path = "demo/temp_test_image.png";
    stbi_write_png(temp_image_path.c_str(), out_width, out_height, out_channels,
                   resized_cropped_image.get(), out_width * out_channels);

    const ImageLoaderConfig resized_config(temp_image_path);

    auto image_tensor = loader.load(resized_config);

    Normalize normalizer;
    auto normalized_tensor = normalizer.normalize(
        image_tensor, {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});

    std::cout << normalized_tensor << std::endl;

    inputs["input"] = normalized_tensor;
    outputs = model->infer(inputs);

  } catch (const std::exception& e) {
    std::cerr << "Inference failed: " << e.what() << '\n';
  }
  if (outputs.empty()) {
    std::cerr << "Error: output empty" << std::endl;
  }

  auto prediction = outputs.find("output");
  auto output_tensor =
      std::get<std::shared_ptr<Tensor<float>>>(prediction->second);
  int max_index = TensorOperations::arg_max<float>(output_tensor);

  std::vector<int> max_indices =
      TensorOperations::top_n_arg_max<float>(output_tensor, 5);

  Profiler::end_timing("Performing Inference");
  return max_indices;
}

/**
 * @brief Load a test test of mnist images and labels.
 * We use this to validate the Lenet model ModularML constructs internally
 */
int main(int argc, char* argv[]) {
  std::string image_directory = "demo/alexnet_demo/ImageNet";

  std::string model_path = argv[1];

  std::string validation_labels =
      "demo/alexnet_demo/ILSVRC2012_validation_ground_truth.json";
  std::unordered_map<std::string, int> image_name_to_index =
      load_labels(validation_labels, "LABEL_KERAS-CAFFE");

  nlohmann::json json_model;
  std::ifstream file(model_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open json file");
    return 1;
  }

  file >> json_model;
  file.close();

  Parser_mml parser;
  std::unique_ptr<Model> model;

  // Try to parse the model provided
  try {
    model = parser.parse(json_model);
    std::cout << "Successfully loaded model" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error occurred when trying to parse json: " << e.what()
              << '\n';
    return 1;
  }

  int top_one_predictions = 0;
  int top_five_predictions = 0;
  int num_validations = 1;

  Profiler::begin_timing("ImageNet testing");
  for (const auto& image : fs::directory_iterator(image_directory)) {
    if (image.is_regular_file()) {
      std::string image_path = image.path().string();
      std::string image_name = image.path().filename();

      int label_index = image_name_to_index[image_name];
      std::cout << "Correct prediction index: " << label_index << std::endl;

      std::vector<int> prediction = run_alexnet_inference(model, image_path);

      int correct_prediction = image_name_to_index[image_name];
      for (int p : prediction) {
        std::cout << "prediction: " << p << std::endl;
      }
      // Update the prediction stats
      if (prediction[0] == correct_prediction) {
        top_one_predictions++;
        top_five_predictions++;
      } else if (std::find(prediction.begin(), prediction.end(),
                           correct_prediction) != prediction.end()) {
        top_five_predictions++;
      }
    }
  }
  Profiler::end_timing("ImageNet testing");

  std::cout << "Top-1 accuracy: "
            << (static_cast<float>(top_one_predictions) /
                static_cast<float>(num_validations)) *
                   100
            << " %" << std::endl;
  std::cout << "Top-5 accuracy: "
            << (static_cast<float>(top_five_predictions) /
                static_cast<float>(num_validations)) *
                   100
            << " %" << std::endl;

  return 0;
}