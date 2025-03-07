#include <gtest/gtest.h>

#include <modularml>

TEST(ModelTest, InferWithGemmNode) {
    using TensorType = Tensor<float>;
    auto A = std::make_shared<TensorType>(std::vector<float>{1, 2, 3, 4}, std::vector<size_t>{2, 2});
    auto B = std::make_shared<TensorType>(std::vector<float>{5, 6, 7, 8}, std::vector<size_t>{2, 2});
    auto Y = std::make_shared<TensorType>(std::vector<float>(4), std::vector<size_t>{2, 2});

    auto gemmNode = std::make_unique<GemmNode<float>>(A, B, Y);

    Model_mml model;
    model.addNode(std::move(gemmNode));

    array_mml<GeneralDataTypes> inputs = {GeneralDataTypes(A), GeneralDataTypes(B)};
    auto outputs = model.infer(inputs);

    auto outputTensor = std::get<std::shared_ptr<TensorType>>(outputs[0]);
    std::vector<float> expected = {19, 22, 43, 50};
    EXPECT_EQ(outputTensor->get_data(), expected);
}

TEST(ModelTest, InferWithReLUNode) {
    using TensorType = Tensor<float>;
    auto X = std::make_shared<TensorType>(std::vector<float>{-1, 2, -3, 4}, std::vector<size_t>{2, 2});
    auto Y = std::make_shared<TensorType>(std::vector<float>(4), std::vector<size_t>{2, 2});

    auto reluNode = std::make_unique<ReLUNode<float>>(X, Y);

    Model_mml model;
    model.addNode(std::move(reluNode));

    array_mml<GeneralDataTypes> inputs = {GeneralDataTypes(X)};
    auto outputs = model.infer(inputs);

    auto outputTensor = std::get<std::shared_ptr<TensorType>>(outputs[0]);
    std::vector<float> expected = {0, 2, 0, 4};
    EXPECT_EQ(outputTensor->get_data(), expected);
}

TEST(ModelTest, InferWithSwishNode) {
    using TensorType = Tensor<float>;
    auto X = std::make_shared<TensorType>(std::vector<float>{-1, 2, -3, 4}, std::vector<size_t>{2, 2});
    auto Y = std::make_shared<TensorType>(std::vector<float>(4), std::vector<size_t>{2, 2});

    auto swishNode = std::make_unique<SwishNode<float>>(X, Y);

    Model_mml model;
    model.addNode(std::move(swishNode));

    array_mml<GeneralDataTypes> inputs = {GeneralDataTypes(X)};
    auto outputs = model.infer(inputs);

    auto outputTensor = std::get<std::shared_ptr<TensorType>>(outputs[0]);
    std::vector<float> expected = {-0.268941, 1.76159, -0.142277, 3.92806}; // Approximate expected values
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(outputTensor->get_data()[i], expected[i], 1e-5);
    }
}

TEST(ModelTest, InferWithTanHNode) {
    using TensorType = Tensor<float>;
    auto X = std::make_shared<TensorType>(std::vector<float>{-1, 2, -3, 4}, std::vector<size_t>{2, 2});
    auto Y = std::make_shared<TensorType>(std::vector<float>(4), std::vector<size_t>{2, 2});

    auto tanhNode = std::make_unique<TanHNode<float>>(X, Y);

    Model_mml model;
    model.addNode(std::move(tanhNode));

    array_mml<GeneralDataTypes> inputs = {GeneralDataTypes(X)};
    auto outputs = model.infer(inputs);

    auto outputTensor = std::get<std::shared_ptr<TensorType>>(outputs[0]);
    std::vector<float> expected = {-0.761594, 0.964028, -0.995055, 0.999329}; // Approximate expected values
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(outputTensor->get_data()[i], expected[i], 1e-5);
    }
}