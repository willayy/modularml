#include <fstream>
#include <gtest/gtest.h>
#include <modularml>

#include "data/parser_values.hpp"

TEST(test_parser_model, test_parsing_and_running_model) {
    std::ifstream file("../test.json");
    ASSERT_TRUE(file.is_open()) << "Failed to open test.json file";

  nlohmann::json onnx_model;
  file >> onnx_model;
  file.close();

  std::unique_ptr<Model> model_base;
  ASSERT_NO_THROW({ model_base = DataParser::parse(onnx_model); })
      << "Parser failed to parse the JSON file";

  ASSERT_NE(model_base, nullptr) << "Model is null";

  auto model = dynamic_cast<Model_mml *>(model_base.get());
  ASSERT_NE(model, nullptr) << "Model is not of type Model_mml";

  std::unordered_map<std::string, GeneralDataTypes> inputs;

  array_mml<size_t> shape({1, 2});
  array_mml<float> values({0.5f, -0.5f});

  auto input_tensor = std::make_shared<Tensor<float>>(shape, values);
  inputs["input"] = input_tensor;

  std::unordered_map<std::string, GeneralDataTypes> outputs;
  ASSERT_NO_THROW({ outputs = model->infer(inputs); })
      << "Inference failed on the parsed model";

  ASSERT_FALSE(outputs.empty()) << "Model produced no outputs";

  auto output_it = outputs.find("output");
  ASSERT_NE(output_it, outputs.end()) << "Expected output tensor not found";

  ASSERT_TRUE(
      std::holds_alternative<std::shared_ptr<Tensor<float>>>(output_it->second))
      << "Output is not a float tensor";

  auto output_tensor =
      std::get<std::shared_ptr<Tensor<float>>>(output_it->second);

  EXPECT_EQ(output_tensor->get_shape(), array_mml<size_t>({1, 3}));

  auto expected_output =
      array_mml<float>({-0.46211716f, -0.46211716f, -0.46211716f});

  for (size_t i = 0; i < output_tensor->get_size(); i++) {
    EXPECT_NEAR(output_tensor->get_data()[i], expected_output[i], 1e-5);
  }
}

TEST(test_parser_model, test_parsing_and_running_lenet) {
  std::ifstream file("../lenet.json");
  ASSERT_TRUE(file.is_open()) << "Failed to open lenet.json file";

  nlohmann::json onnx_model;
  file >> onnx_model;
  file.close();

  std::unique_ptr<Model> model_base;
  ASSERT_NO_THROW({ model_base = DataParser::parse(onnx_model); })
      << "Parser failed to parse the JSON file";

  ASSERT_NE(model_base, nullptr) << "Model is null";

  auto model = dynamic_cast<Model_mml *>(model_base.get());
  ASSERT_NE(model, nullptr) << "Model is not of type Model_mml";

  std::unordered_map<std::string, GeneralDataTypes> inputs;
    
  auto input_tensor = std::make_shared<Tensor<float>>(array_mml<size_t>{INPUT_TENSOR_SHAPE_LENET}, array_mml<float>{INPUT_TENSOR_DATA_LENET});
  inputs["input"] = input_tensor;

  std::unordered_map<std::string, GeneralDataTypes> outputs;
  ASSERT_NO_THROW({ outputs = model->infer(inputs); })
      << "Inference failed on the parsed model";

  ASSERT_FALSE(outputs.empty()) << "Model produced no outputs";

  auto output_it = outputs.find("output");
  ASSERT_NE(output_it, outputs.end()) << "Expected output tensor not found";

  ASSERT_TRUE(
      std::holds_alternative<std::shared_ptr<Tensor<float>>>(output_it->second))
      << "Output is not a float tensor";

  auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);
    
  auto expected_output_tensor = std::make_shared<Tensor<float>>(array_mml<size_t>{OUTPUT_TENSOR_SHAPE_LENET}, array_mml<float>{OUTPUT_TENSOR_DATA_LENET});

  ASSERT_TRUE(TensorUtils::tensors_are_close(*output_tensor, *expected_output_tensor, 0.0125f));
  int max_index =  Arithmetic::arg_max<float>(output_tensor);
  ASSERT_TRUE(max_index == PREDICTED_CLASS_LENET);
}

TEST(test_parser_model, test_parsing_and_running_alexnet) {
    std::ifstream file("../alexnet.json");
    if (!file.is_open()) {
        GTEST_SKIP() << "Skipping test as alexnet.json file is not found";
    }

    nlohmann::json onnx_model;
    file >> onnx_model;
    file.close();

    std::unique_ptr<Model> model_base;
    ASSERT_NO_THROW({
        model_base = DataParser::parse(onnx_model);
    }) << "Parser failed to parse the JSON file";

    ASSERT_NE(model_base, nullptr) << "Model is null";

    auto model = dynamic_cast<Model_mml*>(model_base.get());
    ASSERT_NE(model, nullptr) << "Model is not of type Model_mml";

    std::unordered_map<std::string, GeneralDataTypes> inputs;
    
    auto input_tensor = std::make_shared<Tensor<float>>(array_mml<size_t>{INPUT_TENSOR_SHAPE_ALEX}, array_mml<float>{INPUT_TENSOR_DATA_ALEX});
    inputs["input"] = input_tensor;

    std::unordered_map<std::string, GeneralDataTypes> outputs;
    ASSERT_NO_THROW({
        outputs = model->infer(inputs);
    }) << "Inference failed on the parsed model";

    ASSERT_FALSE(outputs.empty()) << "Model produced no outputs";

    auto output_it = outputs.find("output");
    ASSERT_NE(output_it, outputs.end()) << "Expected output tensor not found";

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<Tensor<float>>>(output_it->second))
        << "Output is not a float tensor";

    auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);

    auto expected_output_tensor = std::make_shared<Tensor<float>>(array_mml<size_t>{OUTPUT_TENSOR_SHAPE_ALEX}, array_mml<float>{OUTPUT_TENSOR_DATA_ALEX});


    //ASSERT_TRUE(TensorUtils::tensors_are_close(*output_tensor, *expected_output_tensor, 0.0125f));
    int max_index = Arithmetic::arg_max<float>(output_tensor);
    ASSERT_TRUE(max_index == PREDICTED_CLASS_ALEX);
}