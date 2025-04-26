#include <fstream>
#include <iostream>
#include <stdexcept>

#include "backend/dataloader/normalizer.hpp"
#include "backend/dataloader/resize_and_cropper.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <modularml>

#include "stb_image_write.h"

std::string get_class_name(const std::string& filename, const std::string& id) {
  std::ifstream file(filename);
  nlohmann::json j;
  file >> j;
  return j[id][1];
}

int run_alexnet_inference(const std::unique_ptr<Model>& model,
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
    return 1;
  }
  if (outputs.empty()) {
    std::cerr << "Error: output empty" << std::endl;
    return 1;
  }

  auto prediction = outputs.find("output");
  auto output_tensor =
      std::get<std::shared_ptr<Tensor<float>>>(prediction->second);
  int max_index = TensorOperations::arg_max<float>(output_tensor);

  std::vector<int> max_indices =
      TensorOperations::top_n_arg_max<float>(output_tensor, 5);

  std::cout << "Prediction: " << std::endl;
  for (int i = 0; i < 5; i++) {
    auto prediction_name =
        get_class_name("demo/alexnet_demo/alexnet_ImageNet_labels.json",
                       std::to_string(max_indices.at(i)));
    float prediction_percentage = (*output_tensor)[max_indices.at(i)];

    std::cout << i + 1 << ": " << prediction_name << "  (" << std::fixed
              << std::setprecision(4) << prediction_percentage << " %)"
              << std::endl;
  }

  Profiler::end_timing("Performing Inference");
  return 0;
}

int parse_alexnet(int argc, char* argv[]) {
  std::string model_path = argv[1];

  std::cout << "model provided: " << model_path << std::endl;

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

  Profiler::begin_timing("Parsing AlexNet");
  // Try to parse the model provided
  try {
    model = parser.parse(json_model);
  } catch (const std::exception& e) {
    std::cerr << "Error occurred when trying to parse json: " << e.what()
              << '\n';
    return 1;
  }
  Profiler::end_timing("Parsing AlexNet");
  return 0;
}

int alexnet_demo(int argc, char* argv[]) {
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv[0]
              << " model.json           # Interactive mode\n";
    std::cerr << "  " << argv[0]
              << " model.json image.png # One-shot prediction\n";
    return 1;
  }

  Profiler::begin_timing("Entire demo");
  std::string model_path = argv[1];

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

  Profiler::begin_timing("Parsing AlexNet");
  // Try to parse the model provided
  try {
    model = parser.parse(json_model);
  } catch (const std::exception& e) {
    std::cerr << "Error occurred when trying to parse json: " << e.what()
              << '\n';
    return 1;
  }
  Profiler::end_timing("Parsing AlexNet");

  // Single prediction
  if (argc == 3) {
    std::string image_path = argv[2];
    run_alexnet_inference(model, image_path);
  }

  // Just the json was provided, enter interactive mode
  else {
    std::string image_path;
    while (true) {
      std::cout << "\nEnter image path (or 'q' to quit): ";
      std::getline(std::cin, image_path);

      if (image_path == "q" || image_path == "quit" || image_path == "exit")
        break;

      if (!std::ifstream(image_path)) {
        std::cerr << "Image file does not exist: " << image_path << '\n';
        continue;
      }

      run_alexnet_inference(model, image_path);
    }
  }
  Profiler::end_timing("Entire demo");

  return 0;
}

int run_lenet_inference(const std::unique_ptr<Model>& model,
                        std::string image_path) {
  Profiler::begin_timing("Performing Inference");

  std::unordered_map<std::string, GeneralDataTypes> inputs;
  std::unordered_map<std::string, GeneralDataTypes> outputs;

  try {
    const ImageLoaderConfig config(image_path);

    ImageLoader loader;

    auto image_tensor = loader.load(config);
    inputs["input"] = image_tensor;
    outputs = model->infer(inputs);

  } catch (const std::exception& e) {
    std::cerr << "Inference failed: " << e.what() << '\n';
    return 1;
  }
  if (outputs.empty()) {
    std::cerr << "Error: output empty" << std::endl;
    return 1;
  }
  // Sequential is something i found during debug seems to be different key with
  // different frameworks that export onnx
  auto prediction = outputs.find("sequential");

  if (prediction == outputs.end()) {
    throw std::runtime_error("Demo: Output tensor not found in iomap");
  }
  auto output_tensor =
      std::get<std::shared_ptr<Tensor<float>>>(prediction->second);
  int max_index = TensorOperations::arg_max<float>(output_tensor);

  std::vector<int> max_indices =
      TensorOperations::top_n_arg_max<float>(output_tensor, 5);

  std::cout << "Prediction: " << std::endl;
  for (int i = 0; i < 5; i++) {
    int prediction_index = max_indices.at(i);
    float prediction_value = (*output_tensor)[max_indices.at(i)];

    std::cout << i + 1 << ": " << prediction_index << "  (" << std::fixed
              << std::setprecision(4) << prediction_value << " %)" << std::endl;
  }

  Profiler::end_timing("Performing Inference");
  return 0;
}

int lenet_demo(int argc, char* argv[]) {
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv[0]
              << " model.json           # Interactive mode\n";
    std::cerr << "  " << argv[0]
              << " model.json image.png # One-shot prediction\n";
    return 1;
  }

  Profiler::begin_timing("Entire LeNet Demo");
  std::string model_path = argv[1];

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

  Profiler::begin_timing("Parsing LeNet");
  // Try to parse the model provided
  try {
    model = parser.parse(json_model);
  } catch (const std::exception& e) {
    std::cerr << "Error occurred when trying to parse json: " << e.what()
              << '\n';
    return 1;
  }
  Profiler::begin_timing("Parsing LeNet");

  // Single prediction
  if (argc == 3) {
    std::string image_path = argv[2];
    run_lenet_inference(model, image_path);
  }
  // Just the json was provided, enter interactive mode
  else {
    std::string image_path;
    while (true) {
      std::cout << "\nEnter image path (or 'q' to quit): ";
      std::getline(std::cin, image_path);

      if (image_path == "q" || image_path == "quit" || image_path == "exit")
        break;

      if (!std::ifstream(image_path)) {
        std::cerr << "Image file does not exist: " << image_path << '\n';
        continue;
      }

      run_alexnet_inference(model, image_path);
    }
  }
  Profiler::end_timing("Entire LeNet Demo");

  return 0;
}

int main(int argc, char* argv[]) {
  lenet_demo(argc, argv);

  return 0;
}
