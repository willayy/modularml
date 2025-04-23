#include <gtest/gtest.h>

#include <fstream>
#include <future>
#include <modularml>
#include <thread>

/**
 * @brief Retrieves the CAFFE label for a given image key from a JSON file.
 *
 * This function reads a JSON file, searches for the specified image key,
 * and extracts the label associated with "CAFFE". If the file cannot be
 * opened, the image key is not found, or no "CAFFE" label exists, an
 * exception is thrown.
 *
 * @param jsonPath The file path to the JSON file containing image labels.
 * @param imageKey The key corresponding to the image in the JSON file.
 * @return The CAFFE label as an integer.
 *
 * @throws std::runtime_error If the JSON file cannot be opened.
 * @throws std::runtime_error If the image key is not found in the JSON file.
 * @throws std::runtime_error If no CAFFE label is found for the given image key.
 */
int get_caffe_label(const std::string& jsonPath, const std::string& imageKey) {
  std::ifstream file(jsonPath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + jsonPath);
  }

  nlohmann::json jsonData;
  file >> jsonData;
  file.close();

  if (!jsonData.contains(imageKey)) {
    throw std::runtime_error("Image key not found in JSON: " + imageKey);
  }

  const auto& labels = jsonData.at(imageKey);
  for (auto it = labels.begin(); it != labels.end(); ++it) {
    if (it.key().find("CAFFE") != std::string::npos) {
      return it.value();
    }
  }

  throw std::runtime_error("No CAFFE label found for image: " + imageKey);
}

/**
 * @brief Pads an integer with leading zeros to a specified width and returns it as a string.
 *
 * This function takes an integer and converts it to a string, ensuring that the resulting
 * string has a minimum width by padding it with leading zeros if necessary.
 *
 * @param num The integer to be padded.
 * @param width The minimum width of the resulting string. Defaults to 8 if not specified.
 * @return A string representation of the integer, padded with leading zeros to the specified width.
 */
std::string padNumber(int num, int width = 8) {
  std::ostringstream ss;
  ss << std::setw(width) << std::setfill('0') << num;
  return ss.str();
}

/**
 * @brief Processes a range of ImageNet images, performs inference using a pre-trained model,
 *        and evaluates the predictions against ground truth labels.
 *
 * @param startingindex The starting index of the images to process (inclusive). Must be > 0 and <= 50000.
 * @param endingindex The ending index of the images to process (inclusive). Must be >= startingindex and <= 50000.
 * @return std::pair<size_t, size_t> A pair containing the number of successful predictions (first)
 *         and the number of failed predictions (second).
 *
 * @throws std::invalid_argument If:
 *         - endingindex < startingindex
 *         - startingindex > 50000
 *         - endingindex > 50000
 *         - startingindex <= 0
 *         - endingindex < 0
 *
 * @details This function performs the following steps for each image in the specified range:
 *          1. Loads the image from the file system.
 *          2. Resizes and crops the image to the required dimensions.
 *          3. Normalizes the image using predefined mean and standard deviation values.
 *          4. Loads the image into a tensor and sets it as input for the model.
 *          5. Runs inference using a pre-trained AlexNet model.
 *          6. Compares the model's prediction with the ground truth label from a JSON file.
 *          7. Tracks the number of successful and failed predictions.
 *
 * @note The function assumes the existence of specific file paths for images and labels:
 *       - Images are located in "../tests/data/imagenet/images/".
 *       - Ground truth labels are in "../tests/data/imagenet/ILSVRC2012_validation_ground_truth.json".
 *       - The AlexNet model is loaded from "../alexnet.json".
 *
 * @warning Ensure that the file paths and required resources are correctly set up before calling this function.
 */
std::pair<size_t, size_t> imageNet(const size_t startingindex, const size_t endingindex) {
  if (endingindex < startingindex) {
    throw std::invalid_argument("Ending index must be larger than starting index");
  } else if (startingindex > 50000) {
    throw std::invalid_argument("Starting index must be less than 50000");
  } else if (endingindex > 50000) {
    throw std::invalid_argument("Ending index must be less than 50000");
  } else if (startingindex <= 0) {
    throw std::invalid_argument("Starting index must be larger than 0");
  } else if (endingindex == 0) {
    throw std::invalid_argument("Ending index must be larger than 0");
  }

  size_t success = 0;
  size_t failure = 0;
  std::string imagePath = "../tests/data/imagenet/images/";
  std::string labelPath = "../tests/data/imagenet/ILSVRC2012_validation_ground_truth.json";
  std::shared_ptr<ImageLoader> loader = std::make_shared<ImageLoader>();
  Parser_mml parser;
  imageResizeAndCropper resizer_and_cropper;
  Normalize normalizer;

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
  for (size_t i = startingindex; i <= endingindex; ++i) {
    // format the string correctly
    std::string imageFile = "ILSVRC2012_val_" + padNumber(i) + ".JPEG";
    std::string imageFilePath = imagePath + imageFile;
    std::cout << "Processing image: " << imageFile << std::endl;

    // resize and crop the image
    const ImageLoaderConfig config(imageFilePath);
    int out_width, out_height, out_channels;
    std::shared_ptr<unsigned char> resized_image = resizer_and_cropper.resize(config, out_width, out_height, out_channels);

    const int crop_size = 224;
    std::shared_ptr<unsigned char> resized_cropped_image =
        resizer_and_cropper.crop(resized_image, out_width, out_height, out_channels, crop_size);

    // Build the raw image buffer
    ImageLoader::RawImageBuffer raw_buffer = {
        resized_cropped_image,
        crop_size, crop_size,  // width, height
        out_channels           // still 3
    };

    // Load into a tensor directly from memory
    auto image_tensor = loader->load(raw_buffer);

    // Normalize the image (?)
    auto normalized_tensor = normalizer.normalize(image_tensor, {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});

    // Set the input for the model
    inputs["input"] = normalized_tensor;

    // Run inference
    outputs = model->infer(inputs);

    // Get the output tensor & run arg_max
    auto output_it = outputs.find("output");
    auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
    int result = TensorOperations::arg_max<float>(output_tensor);

    // Get the class number from the JSON file
    int expected_result = get_caffe_label(labelPath, imageFile);

    // Check if it matches the prediction
    // Increase success or failure
    if (result == expected_result) {
      success++;
    } else {
      failure++;
    }

    // Print the result for each image
    std::cout << "Image: " << imageFile << ", Predicted: " << result << ", Expected: " << expected_result << std::endl;
  }
  return {success, failure};
}

TEST(test_get_caffe_label, get_caffe_label) {
  std::string labelPath = "../tests/data/imagenet/ILSVRC2012_validation_ground_truth.json";
  auto label1 = "ILSVRC2012_val_" + padNumber(1) + ".JPEG";
  auto label2 = "ILSVRC2012_val_" + padNumber(2) + ".JPEG";

  EXPECT_EQ(get_caffe_label(labelPath, label1), 65);
  EXPECT_EQ(get_caffe_label(labelPath, label2), 970);
}

TEST(test_imageNet, imageNet) {
  // Alexnet running imagenet should have a success rate of 57%

  if (!std::ifstream("../alexnet.json").good()) {
    GTEST_SKIP() << "Skipping test as ../alexnet.json is not found.";
  }

  auto result = imageNet(1, 60);

  float success_rate = static_cast<float>(result.first) / (result.first + result.second);

  std::cout << "Success: " << result.first << ", Failure: " << result.second << std::endl;
  std::cout << "Success Rate: " << success_rate * 100 << "%" << std::endl;
  GTEST_LOG_(INFO) << "Success Rate: " << success_rate * 100 << "%";
  EXPECT_GE(success_rate, 0.50f);
}

TEST(test_imageNet, imageNet_multithreaded) {
  if (!std::ifstream("../alexnet.json").good()) {
    GTEST_SKIP() << "Skipping test as ../alexnet.json is not found.";
  }

  const size_t min_interval = 1;
  const size_t max_interval = 60;

  const size_t total_images = max_interval - min_interval;
  const unsigned int num_threads = std::thread::hardware_concurrency();
  const size_t images_per_thread = (total_images + num_threads - 1) / num_threads;  // ceiling division

  std::vector<std::future<std::pair<size_t, size_t>>> futures;

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start = min_interval + t * images_per_thread;
    size_t end = std::min(start + images_per_thread - 1, max_interval);
    if (start > end) continue;  // nothing to process

    futures.push_back(std::async(std::launch::async, imageNet, start, end));
  }

  size_t total_success = 0;
  size_t total_failure = 0;

  for (auto& future : futures) {
    auto [success, failure] = future.get();
    total_success += success;
    total_failure += failure;
  }

  std::cout << "Total Success: " << total_success << ", Total Failure: " << total_failure << std::endl;

  // Assert a rough expected success rate (optional)
  float success_rate = static_cast<float>(total_success) / (total_success + total_failure);
  std::cout << "Success Rate: " << success_rate * 100 << "%" << std::endl;

  EXPECT_GE(success_rate, 0.50f);  // allow for some margin below 57%
}