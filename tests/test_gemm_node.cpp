#include <gtest/gtest.h>

#include <modularml> // Assuming this header includes the definitions for array_mml, Tensor, Tensor_mml, GemmNode, and GeneralDataTypes

TEST(GemmNodeTest, ForwardMultiplication) {
    // Define dimensions: M = 2, K = 3, N = 2.
    array_mml<int> shapeA({2, 3});
    array_mml<int> shapeB({3, 2});
    array_mml<int> shapeY({2, 2});  // Output shape is [M, N]

    // Create tensor A with values: [1, 2, 3, 4, 5, 6] in row-major order.
    Tensor_mml<float> tensorA(shapeA);
    for (int i = 0; i < 6; i++) {
        tensorA[i] = static_cast<float>(i + 1);  // 1, 2, ..., 6.
    }

    // Create tensor B with values: [7, 8, 9, 10, 11, 12] in row-major order.
    Tensor_mml<float> tensorB(shapeB);
    float valuesB[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; i++) {
        tensorB[i] = valuesB[i];
    }

    // Create an output tensor Y with shape [2,2] and initialize to zero.
    Tensor_mml<float> tensorY(shapeY);
    tensorY.fill(0.0f);

    // Wrap each tensor in a shared pointer.
    auto A_ptr = std::make_shared<Tensor_mml<float>>(tensorA);
    auto B_ptr = std::make_shared<Tensor_mml<float>>(tensorB);
    auto Y_ptr = std::make_shared<Tensor_mml<float>>(tensorY);

    // Construct the GemmNode<float>.
    // Here, alpha = 1.0, beta = 0.0, and no transposition is applied.
    GemmNode<float> node(A_ptr, B_ptr, Y_ptr, std::nullopt, 1.0f, 0.0f, 0, 0);

    // Run the forward pass.
    node.forward();

    // Retrieve output from the node.
    array_mml<GeneralDataTypes> outputs = node.getOutputs();
    ASSERT_EQ(outputs.size(), 1u);

    // Extract the shared pointer to the base Tensor type from the variant.
    auto result_base_ptr = std::get<std::shared_ptr<Tensor<float>>>(outputs[0]);

    // Optionally cast it to the concrete type (Tensor_mml<float>) if you need concrete functionality.
    auto result = std::dynamic_pointer_cast<Tensor_mml<float>>(result_base_ptr);
    ASSERT_NE(result, nullptr) << "Output tensor cast to Tensor_mml<float> failed.";

    // Expected result:
    // First row: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58,
    //            1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64.
    // Second row: 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139,
    //             4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154.
    Tensor_mml<float> expected(shapeY);
    expected[0] = 58.0f;
    expected[1] = 64.0f;
    expected[2] = 139.0f;
    expected[3] = 154.0f;

    // Compare each element of the result with the expected value.
    for (int i = 0; i < expected.get_size(); i++) {
        EXPECT_FLOAT_EQ(expected[i], (*result)[i]);
    }
}