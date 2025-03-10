#include <gtest/gtest.h>

#include <modularml>

TEST(test_node, test_ReLU_float) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f});
  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  ReLUNode<float> reluNode(X, Y);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
}

TEST(test_node, test_ReLU_int32) {
  /**
   * @brief Expected Tensor after the ReLU function is applied to each element.
   */
  auto b = tensor_mml_p<int32_t>({3, 3}, {0, 5, 0, 10, 0, 15, 20, 0, 25});

  auto X = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}, {-7, 5, -3, 10, -2, 15, 20, -6, 25}));
  auto Y = make_shared<Tensor_mml<int32_t>>(Tensor_mml<int32_t>({3, 3}));

  ReLUNode<int32_t> reluNode(X, Y);
  reluNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
}

TEST(test_node, test_TanH_float) {
  /**
   * @brief Expected Tensor after the TanH function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.7615941559557649f, 0.0f, 0.7615941559557649f, -0.9640275800758169f, 0.9640275800758169f, -0.9950547536867305f, 0.9950547536867305f, 0.999329299739067f, -0.999329299739067f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  TanHNode<float> tanhNode(X, Y);
  tanhNode.forward();

  // Retrieve the tensor from the shared pointer Y
  ASSERT_EQ(*Y, *b);
}

TEST(test_node, test_Swish_float) {
  /**
   * @brief Expected Tensor after the Swish function is applied to each element.
   */
  auto b = tensor_mml_p<float>({3, 3}, {-0.2689f, 0.0f, 0.7311f, -0.2384f, 1.7616f, -0.1423f, 2.8577f, 3.9281f, -0.0719f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, 4.0f, -4.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  SwishNode<float> swishNode(X, Y);
  swishNode.forward();

  ASSERT_TRUE(tensors_are_close(*Y, *b));
}

TEST(test_node, test_Softmax_1D) {
  /**
   * @brief Expected Tensor after the Softmax function is applied.
   */
  auto b = tensor_mml_p<float>({5}, {0.0117f, 0.1172f, 0.3160f, 0.8659f, 1.0000f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({5}));

  SoftmaxNode<float> softmaxNode(X, Y, 0);
  softmaxNode.forward();

  // Debug print: Output tensor after softmax
  std::cout << "Computed Softmax Output (1D):\n";
  for (int i = 0; i < 5; i++) {
    std::cout << (*Y)[i] << " ";
  }
  std::cout << std::endl;

  // Ensure the sum is approximately 1
  float sum = (*Y)[0] + (*Y)[1] + (*Y)[2] + (*Y)[3] + (*Y)[4];
  std::cout << "Row sum (1D): " << sum << std::endl;
  ASSERT_NEAR(sum, 1.0f, 1e-5);

  ASSERT_TRUE(tensors_are_close(*Y, *b));
}

TEST(test_node, test_Softmax_2D) {
  /**
   * @brief Expected Tensor after the Softmax function is applied.
   */
  auto b = tensor_mml_p<float>({3, 3}, 
                                {0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}, 
                                {-1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({3, 3}));

  SoftmaxNode<float> softmaxNode(X, Y, 1);
  softmaxNode.forward();

  // Debug print: Output tensor after softmax
  std::cout << "Computed Softmax Output (2D):\n";
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << (*Y)[i * 3 + j] << " ";
    }
    std::cout << std::endl;
  }

  // Ensure each row sums to 1
  int cols = Y->get_shape()[1];
  for (int i = 0; i < 3; i++) {
    float sum = (*Y)[i * cols + 0] + (*Y)[i * cols + 1] + (*Y)[i * cols + 2];

    // Debug print: row sum
    std::cout << "Row " << i << " sum (2D): " << sum << std::endl;

    ASSERT_NEAR(sum, 1.0f, 1e-5);
  }

  ASSERT_TRUE(tensors_are_close(*Y, *b));
}

TEST(test_node, test_Softmax_3D) {
  /**
   * @brief Expected Tensor after the Softmax function is applied.
   */
  auto b = tensor_mml_p<float>({2, 3, 3}, 
                                {0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3, 3}, 
                                {-1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({2, 3, 3}));

  SoftmaxNode<float> softmaxNode(X, Y, 2);
  softmaxNode.forward();

  // Debug print: Output tensor after softmax
  std::cout << "Computed Softmax Output (3D):\n";
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        std::cout << (*Y)[i * 9 + j * 3 + k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // Ensure each slice sums to 1
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      float sum = (*Y)[i * 9 + j * 3 + 0] + (*Y)[i * 9 + j * 3 + 1] + (*Y)[i * 9 + j * 3 + 2];

      // Debug print: row sum
      std::cout << "Slice " << i << " Row " << j << " sum (3D): " << sum << std::endl;

      ASSERT_NEAR(sum, 1.0f, 1e-5);
    }
  }

  ASSERT_TRUE(tensors_are_close(*Y, *b));
}

TEST(test_node, test_Softmax_4D) {
  /**
   * @brief Expected Tensor after the Softmax function is applied.
   */
  auto b = tensor_mml_p<float>({1, 2, 3, 3}, 
                                {0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({1, 2, 3, 3}, 
                                {-1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({1, 2, 3, 3}));

  SoftmaxNode<float> softmaxNode(X, Y, 3);
  softmaxNode.forward();

  // Debug print: Output tensor after softmax
  std::cout << "Computed Softmax Output (4D):\n";
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          std::cout << (*Y)[i * 18 + j * 9 + k * 3 + l] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  // Ensure each slice sums to 1
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      float sum = (*Y)[i * 18 + j * 9 + 0] + (*Y)[i * 18 + j * 9 + 1] + (*Y)[i * 18 + j * 9 + 2];

      // Debug print: row sum
      std::cout << "Slice " << i << " Row " << j << " sum (4D): " << sum << std::endl;

      ASSERT_NEAR(sum, 1.0f, 1e-5);
    }
  }

  ASSERT_TRUE(tensors_are_close(*Y, *b));
}

TEST(test_node, test_Softmax_5D) {
  /**
   * @brief Expected Tensor after the Softmax function is applied.
   */
  auto b = tensor_mml_p<float>({1, 1, 2, 3, 3}, 
                                {0.0900f, 0.2447f, 0.6652f, 
                                 0.0900f, 0.2447f, 0.6652f});

  auto X = make_shared<Tensor_mml<float>>(Tensor_mml<float>({1, 1, 2, 3, 3}, 
                                {-1.0f, 0.0f, 1.0f, 
                                 -1.0f, 0.0f, 1.0f}));
  auto Y = make_shared<Tensor_mml<float>>(Tensor_mml<float>({1, 1, 2, 3, 3}));

  SoftmaxNode<float> softmaxNode(X, Y, 4);
  softmaxNode.forward();

  // Debug print: Output tensor after softmax
  std::cout << "Computed Softmax Output (5D):\n";
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 2; k++) {
        for (int l = 0; l < 3; l++) {
          for (int m = 0; m < 3; m++) {
            std::cout << (*Y)[i * 18 + j * 9 + k * 3 + l * 3 + m] << " ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
    }
  }

  // Ensure each slice sums to 1
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 2; k++) {
        float sum = (*Y)[i * 18 + j * 9 + k * 3 + 0] + (*Y)[i * 18 + j * 9 + k * 3 + 1] + (*Y)[i * 18 + j * 9 + k * 3 + 2];

        // Debug print: row sum
        std::cout << "Slice " << i << " Row " << j << " Channel " << k << " sum (5D): " << sum << std::endl;

        ASSERT_NEAR(sum, 1.0f, 1e-5);
      }
    }
  }

  ASSERT_TRUE(tensors_are_close(*Y, *b));
}
