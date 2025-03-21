/**
 * @file test_LeNet.cpp
 * @brief Unit tests for the LeNet model using Google Test framework.
 */

#include <gtest/gtest.h>

#include "test_LeNet.hpp"
#include <modularml>

/**
 * @class LeNetModel
 * @brief Implementation of the LeNet model.
 *
 * This is by no means an example of how models should be implemented
 * using the framework. It is just a way, that has been done in order
 * to mimic Pytorch as well as possible.
 *
 * @tparam T Data type of the tensors.
 */
template <typename T>
class LeNetModel {
 private:
  shared_ptr<Tensor<T>> input_tensor;
  shared_ptr<Tensor<T>> output_tensor;

  shared_ptr<Tensor_mml<float>> W1, W2, W3, W_gemm1, W_gemm2, B_gemm1, B_gemm2;
  shared_ptr<Tensor_mml<T>> conv_output_tensor1, conv_output_tensor2, conv_output_tensor3;
  shared_ptr<Tensor_mml<T>> output_gemm1, output_gemm2;
  shared_ptr<Tensor_mml<T>> output_tensor_ReLU, output_tensor_MaxPool, output_tensor_Reshape, output_tensor_Reshape2, output_tensor_logSoftMax, output_tensor_add;

  unique_ptr<ConvNode<T>> conv1, conv2, conv3;
  unique_ptr<ReLUNode<T>> reluNode;
  unique_ptr<GemmNode<T>> gemm1, gemm2;
  unique_ptr<MaxPoolingNode_mml<T>> maxPoolNode;
  unique_ptr<AddNode<T>> addNode;
  unique_ptr<reshapeNode<T>> reshapeNode1, reshapeNode2;
  unique_ptr<LogSoftMaxNode<T>> logSoftMaxNode;

 public:
  LeNetModel(shared_ptr<Tensor<T>> tensor) {
    this->input_tensor = tensor;

    array_mml<int> tensor_shape = array_mml<int>({1, 1, 32, 32});
    array_mml<float> tensor_data = array_mml<float>(vector<float>(32 * 32, 1.0f));
    this->output_tensor = make_shared<Tensor_mml<T>>(tensor_shape, tensor_data);

    array_mml<int> conv_shape = array_mml<int>({1, 1, 28, 28});
    conv_output_tensor1 = make_shared<Tensor_mml<T>>(conv_shape);
    conv_output_tensor2 = make_shared<Tensor_mml<T>>(conv_shape);
    conv_output_tensor3 = make_shared<Tensor_mml<T>>(conv_shape);

    // Output tensor constructors:
    // Allocate minimal output tensors (single-element tensors)
    // Gemm needs matrixes due to a constraint in the current implementation
    output_gemm1 = make_shared<Tensor_mml<T>>(array_mml<int>{1, 84});
    output_gemm2 = make_shared<Tensor_mml<T>>(array_mml<int>{1, 84});
    output_tensor_ReLU = make_shared<Tensor_mml<T>>(array_mml<int>{1});
    output_tensor_MaxPool = make_shared<Tensor_mml<T>>(array_mml<int>{1});
    output_tensor_Reshape = make_shared<Tensor_mml<T>>(array_mml<int>{1});
    output_tensor_Reshape2 = make_shared<Tensor_mml<T>>(array_mml<int>{1});
    output_tensor_logSoftMax = make_shared<Tensor_mml<T>>(array_mml<int>{1});
    output_tensor_add = make_shared<Tensor_mml<T>>(array_mml<int>{1});

    // Weights generated the same way that they are in pytorch
    std::mt19937 gen(42);
    W1 = std::make_shared<Tensor_mml<T>>(array_mml<int>{6, 1, 5, 5});
    kaimingUniform(W1, 1, 5, gen);

    W2 = std::make_shared<Tensor_mml<T>>(array_mml<int>{16, 6, 5, 5});
    kaimingUniform(W2, 6, 5, gen);

    W3 = std::make_shared<Tensor_mml<T>>(array_mml<int>{120, 16, 5, 5});
    kaimingUniform(W3, 16, 5, gen);

    W_gemm1 = std::make_shared<Tensor_mml<T>>(array_mml<int>{480, 84});
    kaimingUniform(W_gemm1, 120, 480, gen);

    W_gemm2 = std::make_shared<Tensor_mml<T>>(array_mml<int>{84, 10});
    kaimingUniform(W_gemm2, 84, 84, gen);

    // Bias for Gemm
    //B_gemm1 = std::make_shared<Tensor_mml<T>>(array_mml<int>{1, 84});
    array_mml<int> bias1_shape = array_mml<int>{BIAS1_SHAPE};
    array_mml<float> bias1_data = array_mml<float>{BIAS1_DATA};
    B_gemm1 = std::make_shared<Tensor_mml<float>>(bias1_shape, bias1_data);
    array_mml<int> bias2_shape = array_mml<int>{BIAS2_SHAPE};
    array_mml<float> bias2_data = array_mml<float>{BIAS2_DATA};
    B_gemm2 = std::make_shared<Tensor_mml<float>>(bias2_shape, bias2_data);

    // Convolutional layers
    conv1 = make_unique<ConvNode<T>>(input_tensor, W1, conv_output_tensor1,
                                     array_mml<int>{1, 1},        // dilation = 1
                                     array_mml<int>{2, 2, 2, 2},  // padding = 2
                                     array_mml<int>{5, 5},        // kernel size = 5
                                     array_mml<int>{1, 1},        // stride = 1
                                     std::nullopt,                // No bias
                                     1);                          // groups = 1

    conv2 = make_unique<ConvNode<T>>(input_tensor, W2, conv_output_tensor2,
                                     array_mml<int>{1, 1},        // dilation = 1
                                     array_mml<int>{0, 0, 0, 0},  // padding = 0 (default)
                                     array_mml<int>{5, 5},        // kernel size = 5
                                     array_mml<int>{1, 1},        // stride = 1
                                     std::nullopt,                // No bias
                                     1);                          // groups = 1

    conv3 = make_unique<ConvNode<T>>(input_tensor, W3, conv_output_tensor3,
                                     array_mml<int>{1, 1},        // dilation = 1
                                     array_mml<int>{0, 0, 0, 0},  // padding = 0 (default)
                                     array_mml<int>{5, 5},        // kernel size = 5
                                     array_mml<int>{1, 1},        // stride = 1
                                     std::nullopt,                // No bias
                                     1);                          // groups = 1

    // Gemm layers
    gemm1 = make_unique<GemmNode<T>>(
        input_tensor,  // A (Input tensor of shape (batch_size, 480))
        W_gemm1,       // B (Weight tensor of shape (480, 84))
        output_gemm1,  // Y (Output tensor of shape (batch_size, 84))
        B_gemm1,       // C (Bias tensor of shape (84,))
        1.0f,          // alpha (default 1.0)
        1.0f,          // beta (default 1.0)
        0,             // transA = 0 (No transpose for input)
        1              // transB = 1 (Transpose weight matrix)
    );

    gemm2 = make_unique<GemmNode<T>>(
        input_tensor,  // A (Input tensor of shape (batch_size, 84))
        W_gemm2,       // B (Weight tensor of shape (84, 10))
        output_gemm2,  // Y (Output tensor of shape (batch_size, 10))
        B_gemm2,       // C (Bias tensor of shape (10,))
        1.0f,          // alpha (default 1.0)
        1.0f,          // beta (default 1.0)
        0,             // transA = 0 (No transpose for input)
        1              // transB = 1 (Transpose weight matrix)
    );

    // Other layers
    reluNode = make_unique<ReLUNode<T>>(input_tensor, output_tensor_ReLU);
    maxPoolNode = make_unique<MaxPoolingNode_mml<T>>(vector<int>{2, 2}, vector<int>{2, 2}, input_tensor, "VALID");
    addNode = make_unique<AddNode<T>>(input_tensor, input_tensor, output_tensor_add);
    logSoftMaxNode = make_unique<LogSoftMaxNode<T>>(input_tensor, output_tensor_logSoftMax);
  }

  shared_ptr<Tensor<T>> getTensor() {
    return output_tensor;
  }

  void forward() {
    // Convolution 1
    conv1->forward();
    *input_tensor = *conv_output_tensor1;

    // Relu
    reluNode->forward();
    *input_tensor = *output_tensor_ReLU;
    std::cout << "Input shape after conv1 and relu is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 6, 32, 32}) {
      throw std::runtime_error("Shape of input tensor after conv1 is not correct.");
    }

    // Max Pooling, works a bit different than the other nodes atm
    maxPoolNode->forward();
    *output_tensor_MaxPool = *std::get<std::shared_ptr<Tensor<T>>>(maxPoolNode->getOutputs()[0]);
    *input_tensor = *output_tensor_MaxPool;
    std::cout << "Input shape after maxpool1 is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 6, 16, 16}) {
      throw std::runtime_error("Shape of input tensor after maxpool1 is not correct.");
    }

    // Convolution 2
    conv2->forward();
    *input_tensor = *conv_output_tensor2;

    // Relu
    reluNode->forward();
    *input_tensor = *output_tensor_ReLU;
    std::cout << "Input shape after conv2 is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 16, 12, 12}) {
      throw std::runtime_error("Shape of input tensor after conv2 is not correct.");
    }

    // Max Pooling
    maxPoolNode->forward();
    *output_tensor_MaxPool = *std::get<std::shared_ptr<Tensor<T>>>(maxPoolNode->getOutputs()[0]);
    *input_tensor = *output_tensor_MaxPool;
    std::cout << "Input shape after maxpool2 is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 16, 6, 6}) {
      throw std::runtime_error("Shape of input tensor after maxpool2 is not correct.");
    }

    // Add
    // copy the tensor
    auto copytensor = make_shared<Tensor_mml<T>>(*output_tensor_MaxPool);
    // Update the inputs
    array_mml<GeneralDataTypes> inputs({copytensor, input_tensor});
    addNode->setInputs(inputs);
    // Add
    addNode->forward();
    *input_tensor = *output_tensor_add;
    std::cout << "Input shape after add is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 16, 6, 6}) {
      throw std::runtime_error("Shape of input tensor after add is not correct.");
    }

    // Convolution 3
    conv3->forward();
    *input_tensor = *conv_output_tensor3;

    // Relu
    reluNode->forward();
    *input_tensor = *output_tensor_ReLU;
    std::cout << "Input shape after conv3 is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 120, 2, 2}) {
      throw std::runtime_error("Shape of input tensor after conv3 is not correct.");
    }

    // Flatten - declared inside the forward function
    auto batch_size = input_tensor->get_shape()[0];
    auto shape_array = array_mml<int>{2};  // Define the shape of the shape tensor
    auto shape_data = array_mml<int64_t>{batch_size, -1};  // Reshape values
    auto shape_tensor = make_shared<Tensor_mml<int64_t>>(shape_array, shape_data);
    
    reshapeNode1 = make_unique<reshapeNode<T>>(input_tensor, shape_tensor, output_tensor_Reshape);
    reshapeNode1->forward();
    *input_tensor = *output_tensor_Reshape;
    std::cout << "Input shape after reshape1 is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 480}) {
      throw std::runtime_error("Shape of input tensor after reshape1 is not correct.");
    }

    std::cout << "Shape of W_gemm1 before gemm1: " << W_gemm1->get_shape() << std::endl;
    std::cout << "Shape of input_tensor before Gemm1: " << input_tensor->get_shape() << std::endl;
    std::cout << "Shape of B_gemm1 before Gemm1: " << B_gemm1->get_shape() << std::endl;
    // Gemm 1
    gemm1->forward();
    *input_tensor = *output_gemm1;
    // Relu
    reluNode->forward();
    *input_tensor = *output_tensor_ReLU;
    std::cout << "Input shape after gemm1 is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 84}) {
      throw std::runtime_error("Shape of input tensor after gemm1 is not correct.");
    }

    // Gemm 2
    gemm2->forward();
    *input_tensor = *output_gemm2;
    std::cout << "Input shape after gemm2 is: " << input_tensor->get_shape() << std::endl;
    if (input_tensor->get_shape() != array_mml<int>{1, 10}) {
      throw std::runtime_error("Shape of input tensor after gemm2 is not correct.");
    }

    // LogSoftMax
    logSoftMaxNode->forward();
    // FINAL RESULT
    *output_tensor = *output_tensor_logSoftMax;
    std::cout << "Final shape after logSoftMax is: " << output_tensor->get_shape() << std::endl;
    if (output_tensor->get_shape() != array_mml<int>{1, 10}) {
      throw std::runtime_error("Shape of output tensor after logSoftMax is not correct.");
    }

  }

};

/**
 * @brief Test the forward pass of the LeNet model.
 * Later on this will make use of results from Pytorch and compare them.
 * I just want to make sure that the forward pass is working first.
 */
TEST(test_LeNet, test_LeNet_forward) {
  auto TensorToProcess = tensor_mml_p<float>({INPUTTENSOR_SHAPE}, {INPUTTENSOR_DATA});
  LeNetModel<float> model(TensorToProcess);

  model.forward();
  auto output_tensor = model.getTensor();

  auto expected_output = tensor_mml_p<float>({EXPECTED_SHAPE}, {EXPECTED_DATA});
  std::cout << "Output tensor: " << *output_tensor << std::endl;

  ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_output, 0.05f));
  //ASSERT_EQ(*output_tensor, *expected_output);
}
