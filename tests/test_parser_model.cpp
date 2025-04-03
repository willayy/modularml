#include <gtest/gtest.h>
#include <fstream>
#include <modularml>

TEST(test_mml_tensor, test_parsing_and_running_model) {
    std::ifstream file("../test.json");
    ASSERT_TRUE(file.is_open()) << "Failed to open test.json file";

    json onnx_model;
    file >> onnx_model;
    file.close();

    Parser_mml parser;

    std::unique_ptr<Model> model_base;
    ASSERT_NO_THROW({
        model_base = parser.parse(onnx_model);
    }) << "Parser failed to parse the JSON file";

    ASSERT_NE(model_base, nullptr) << "Model is null";

    auto model = dynamic_cast<Model_mml*>(model_base.get());
    ASSERT_NE(model, nullptr) << "Model is not of type Model_mml";

    std::unordered_map<string, GeneralDataTypes> inputs;

    array_mml<uli> shape({1, 2});
    array_mml<float> values({0.5f, -0.5f});


    auto input_tensor = std::make_shared<Tensor_mml<float>>(shape, values);
    inputs["input"] = input_tensor;

    std::unordered_map<string, GeneralDataTypes> outputs;
    ASSERT_NO_THROW({
        outputs = model->infer(inputs);
    }) << "Inference failed on the parsed model";

    ASSERT_FALSE(outputs.empty()) << "Model produced no outputs";

    auto output_it = outputs.find("output");
    ASSERT_NE(output_it, outputs.end()) << "Expected output tensor not found";

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<Tensor<float>>>(output_it->second))
        << "Output is not a float tensor";

    auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);

    EXPECT_EQ(output_tensor->get_shape(), array_mml<uli>({1, 3}));

    // Dynamic cast to Tensor_mml<float> to access the data
    auto output_tensor_mml = std::dynamic_pointer_cast<Tensor_mml<float>>(output_tensor);
    ASSERT_NE(output_tensor_mml, nullptr) << "Failed to cast to Tensor_mml<float>";

    auto expected_output = array_mml<float>({-0.46211716f, -0.46211716f, -0.46211716f});

    for (uli i = 0; i < output_tensor_mml->get_size(); i++) {
        EXPECT_NEAR(output_tensor_mml->get_data()[i], expected_output[i], 1e-5);
    }
}

TEST(test_mml_tensor, test_parsing_and_running_lenet) {
    std::ifstream file("../lenet.json");
    ASSERT_TRUE(file.is_open()) << "Failed to open lenet.json file";

    json onnx_model;
    file >> onnx_model;
    file.close();

    Parser_mml parser;

    std::unique_ptr<Model> model_base;
    ASSERT_NO_THROW({
        model_base = parser.parse(onnx_model);
    }) << "Parser failed to parse the JSON file";

    ASSERT_NE(model_base, nullptr) << "Model is null";

    auto model = dynamic_cast<Model_mml*>(model_base.get());
    ASSERT_NE(model, nullptr) << "Model is not of type Model_mml";

    std::unordered_map<string, GeneralDataTypes> inputs;

    // Create a shape for LeNet: [1, 1, 32, 32] - batch size, channels, height, width
    array_mml<uli> shape({1, 1, 32, 32});
    
    // Create 32x32=1024 values
    std::vector<float> data_vec(1024, 0.0f);  // Initialize all to 0
    
    // Fill with a simple pattern (optional)
    for (size_t i = 0; i < data_vec.size(); ++i) {
        data_vec[i] = static_cast<float>(i % 256) / 255.0f;  // Normalized values between 0-1
    }
    
    array_mml<float> values(data_vec);
    
    std::cout << "Input tensor shape: [1, 1, 32, 32], total elements: " << values.size() << std::endl;
    
    auto input_tensor = std::make_shared<Tensor_mml<float>>(shape, values);
    inputs["input"] = input_tensor;

    std::unordered_map<string, GeneralDataTypes> outputs;
    ASSERT_NO_THROW({
        outputs = model->infer(inputs);
    }) << "Inference failed on the parsed model";

    ASSERT_FALSE(outputs.empty()) << "Model produced no outputs";

    auto output_it = outputs.find("output");
    ASSERT_NE(output_it, outputs.end()) << "Expected output tensor not found";

    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<Tensor<float>>>(output_it->second))
        << "Output is not a float tensor";

    auto output_tensor = std::get<std::shared_ptr<Tensor<float>>>(output_it->second);

    // NOTE: Update this to match the expected LeNet output shape (likely [1, 10] for digit classification)
    // EXPECT_EQ(output_tensor->get_shape(), array_mml<uli>({1, 10}));

    // Dynamic cast to Tensor_mml<float> to access the data
    auto output_tensor_mml = std::dynamic_pointer_cast<Tensor_mml<float>>(output_tensor);
    ASSERT_NE(output_tensor_mml, nullptr) << "Failed to cast to Tensor_mml<float>";

    // Print the output shape and values for debugging
    std::cout << "Output tensor shape: [";
    for (size_t i = 0; i < output_tensor_mml->get_shape().size(); ++i) {
        std::cout << output_tensor_mml->get_shape()[i];
        if (i < output_tensor_mml->get_shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Output values: ";
    for (size_t i = 0; i < std::min(output_tensor_mml->get_size(), static_cast<uli>(10)); ++i) {
        std::cout << output_tensor_mml->get_data()[i] << " ";
    }
    std::cout << std::endl;
    
    // For now, let's just check that values aren't NaN or infinity
    for (uli i = 0; i < output_tensor_mml->get_size(); ++i) {
        EXPECT_FALSE(std::isnan(output_tensor_mml->get_data()[i])) 
            << "Output contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(output_tensor_mml->get_data()[i])) 
            << "Output contains infinity at index " << i;
    }
    
    // Once you know the expected output, add specific assertions
    // auto expected_output = array_mml<float>({value1, value2, ...});
    // for (uli i = 0; i < output_tensor_mml->get_size(); ++i) {
    //     EXPECT_NEAR(output_tensor_mml->get_data()[i], expected_output[i], 1e-5);
    // }
}