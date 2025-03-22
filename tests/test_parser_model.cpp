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

    array_mml<int> shape({1, 2});
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

    EXPECT_EQ(output_tensor->get_shape(), array_mml<int>({1, 3}));

    // Dynamic cast to Tensor_mml<float> to access the data
    auto output_tensor_mml = std::dynamic_pointer_cast<Tensor_mml<float>>(output_tensor);
    ASSERT_NE(output_tensor_mml, nullptr) << "Failed to cast to Tensor_mml<float>";

    auto expected_output = array_mml<float>({-0.46211716f, -0.46211716f, -0.46211716f});

    for (int i = 0; i < output_tensor_mml->get_size(); i++) {
        EXPECT_NEAR(output_tensor_mml->get_data()[i], expected_output[i], 1e-5);
    }
}