#include <gtest/gtest.h>

#include <fstream>
#include <modularml>

#include "stb_image_write.h"

/**
 * @brief Reads the class name corresponding to the given index from a JSON labels file.
 *
 * @param labels_file Path to the JSON file containing label mappings.
 * @param idx         The stringified class index to look up.
 * @return The human-readable class name.
 */
static std::string get_class_name(const std::string& labels_file,
                                  const std::string& idx) {
  std::ifstream f(labels_file);
  nlohmann::json j;
  f >> j;
  return j[idx][1];
}

/**
 * @brief Loads an image, resizes and center-crops it to 224×224, writes it temporarily to disk,
 *        then loads it back as a tensor and normalizes it using ImageNet statistics.
 *
 * @param image_path Path to the input image file.
 * @return A normalized [1×3×224×224] tensor ready for model input.
 */
static std::shared_ptr<Tensor<float>> load_and_preprocess(
    const std::string& image_path) {
  // Resize
  ImageLoaderConfig cfg(image_path);
  imageResizeAndCropper resizer;
  int W, H, C;
  auto raw = resizer.resize(cfg, W, H, C);

  // Center-crop to 224×224
  const int crop = 224;
  auto cropped = resizer.crop(raw, W, H, C, crop);
  W = H = crop;

  // Write to temp PNG and reload via ImageLoader
  const char* tmp = "../tests/data/temp.png";
  stbi_write_png(tmp, W, H, C, cropped.get(), W * C);
  auto loader = std::make_shared<ImageLoader>();
  auto img_tensor = loader->load(ImageLoaderConfig(tmp));
  std::remove(tmp);

  // Normalize with ImageNet mean/std
  Normalize norm;
  return norm.normalize(img_tensor,
                        {0.485f, 0.456f, 0.406f},
                        {0.229f, 0.224f, 0.225f});
}

/**
 * @brief Runs inference on the provided input tensor using a parsed ONNX model,
 *        then returns the predicted class name by taking argmax and looking it up.
 *
 * @param input         The preprocessed input tensor.
 * @param model_json    Path to the ONNX-to-JSON converted model file.
 * @param labels_json   Path to the ImageNet labels JSON file.
 * @return The predicted class name.
 * @throws std::runtime_error if the model JSON cannot be opened or parsed.
 */
static std::string infer_and_get_class(const std::shared_ptr<Tensor<float>>& input,
                                       const std::string& model_json,
                                       const std::string& labels_json) {
  // load & parse JSON model (no skipping here!)
  std::ifstream mf(model_json);
  if (!mf.is_open()) {
    throw std::runtime_error("Cannot open model JSON: " + model_json);
  }
  nlohmann::json j;
  mf >> j;
  mf.close();

  Parser_mml parser;
  auto mb = parser.parse(j);  // may throw or assert
  auto model = dynamic_cast<Model_mml*>(mb.get());

  std::unordered_map<std::string, GeneralDataTypes> in{{"input", input}}, out;
  out = model->infer(in);

  auto tptr = std::get<std::shared_ptr<Tensor<float>>>(out.at("output"));
  int idx = TensorOperations::arg_max<float>(tptr);
  return get_class_name(labels_json, std::to_string(idx));
}

TEST(AlexNet, Foxhound) {
  const std::string model_json = "../alexnet.json";
  const std::string labels_json = "../tests/data/alexnet/alexnet_ImageNet_labels.json";

  if (!std::filesystem::exists(model_json)) {
    GTEST_SKIP() << "Skipping because model JSON not found: " << model_json;
  }

  auto tensor = load_and_preprocess(
      "../tests/data/alexnet/alexnet_pictures/foxhound.png");
  std::string cls;
  ASSERT_NO_THROW(cls = infer_and_get_class(tensor, model_json, labels_json));
  EXPECT_EQ(cls, "English_foxhound");
}

TEST(AlexNet, Egret) {
  const std::string model_json = "../alexnet.json";
  const std::string labels_json = "../tests/data/alexnet/alexnet_ImageNet_labels.json";

  if (!std::filesystem::exists(model_json)) {
    GTEST_SKIP() << "Skipping because model JSON not found: " << model_json;
  }

  auto tensor = load_and_preprocess(
      "../tests/data/alexnet/alexnet_pictures/American_egret.png");
  std::string cls;
  ASSERT_NO_THROW(cls = infer_and_get_class(tensor, model_json, labels_json));
  EXPECT_EQ(cls, "American_egret");
}

TEST(ResNet18, Foxhound) {
  const std::string model_json = "../resnet18.json";
  const std::string labels_json = "../tests/data/alexnet/alexnet_ImageNet_labels.json";

  if (!std::filesystem::exists(model_json)) {
    GTEST_SKIP() << "Skipping because model JSON not found: " << model_json;
  }

  auto tensor = load_and_preprocess(
      "../tests/data/alexnet/alexnet_pictures/foxhound.png");
  std::string cls;
  ASSERT_NO_THROW(cls = infer_and_get_class(tensor, model_json, labels_json));
  EXPECT_EQ(cls, "English_foxhound");
}
