/**
 * @file test_LeNet.cpp
 * @brief Unit tests for the LeNet model using Google Test framework.
 */

#include "test_LeNet.hpp"

#include <gtest/gtest.h>

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
  std::unordered_map<std::string, GeneralDataTypes> iomap;

 public:
  LeNetModel(shared_ptr<Tensor<T>> tensor, bool PytorchWeights = false) {
    // Input tensor
    iomap["input"] = tensor;

    // Output tensor
    auto output_tensor = make_shared<Tensor_mml<T>>(array_mml<uli>{1, 10});
    iomap["output"] = output_tensor;

    // Weights
    if (!PytorchWeights) {
      std::mt19937 gen(42);
      auto W1 = make_shared<Tensor_mml<T>>(array_mml<uli>{6, 1, 5, 5});
      kaiming_uniform(std::static_pointer_cast<Tensor<double>>(W1), 1, 5, gen);  // I have to cast it to double because of
      iomap["W1"] = W1;                                                         // how the function is implemented.

      auto W2 = make_shared<Tensor_mml<T>>(array_mml<uli>{16, 6, 5, 5});
      kaiming_uniform(std::static_pointer_cast<Tensor<double>>(W2), 6, 5, gen);
      iomap["W2"] = W2;

      auto W3 = make_shared<Tensor_mml<T>>(array_mml<uli>{120, 16, 5, 5});
      kaiming_uniform(std::static_pointer_cast<Tensor<double>>(W3), 16, 5, gen);
      iomap["W3"] = W3;

      auto W_gemm1 = make_shared<Tensor_mml<T>>(array_mml<uli>{480, 84});
      kaiming_uniform(std::static_pointer_cast<Tensor<double>>(W_gemm1), 120, 480, gen);
      iomap["W_gemm1"] = W_gemm1;

      auto W_gemm2 = make_shared<Tensor_mml<T>>(array_mml<uli>{84, 10});
      kaiming_uniform(std::static_pointer_cast<Tensor<double>>(W_gemm2), 84, 84, gen);
      iomap["W_gemm2"] = W_gemm2;
    }
    else{
      // Weights
      auto W1 = make_shared<Tensor_mml<double>>(array_mml<uli>{CONV1_WEIGHT_SHAPE}, array_mml<double>{CONV1_WEIGHT_DATA});
      iomap["W1"] = W1;

      auto W2 = make_shared<Tensor_mml<double>>(array_mml<uli>{CONV2_WEIGHT_SHAPE}, array_mml<double>{CONV2_WEIGHT_DATA});
      iomap["W2"] = W2;

      auto W3 = make_shared<Tensor_mml<double>>(array_mml<uli>{CONV3_WEIGHT_SHAPE}, array_mml<double>{CONV3_WEIGHT_DATA});
      iomap["W3"] = W3;

      auto W_gemm1 = make_shared<Tensor_mml<double>>(array_mml<uli>{FC1_WEIGHT_SHAPE}, array_mml<double>{FC1_WEIGHT_DATA});
      iomap["W_gemm1"] = W_gemm1;

      auto W_gemm2 = make_shared<Tensor_mml<double>>(array_mml<uli>{FC2_WEIGHT_SHAPE}, array_mml<double>{FC2_WEIGHT_DATA});
      iomap["W_gemm2"] = W_gemm2;
    }
    // Bias
    auto B_gemm1 = make_shared<Tensor_mml<double>>(array_mml<uli>{FC1_BIAS_SHAPE}, array_mml<double>{FC1_BIAS_DATA});
    iomap["B_gemm1"] = B_gemm1;

    auto B_gemm2 = make_shared<Tensor_mml<double>>(array_mml<uli>{FC2_BIAS_SHAPE}, array_mml<double>{FC2_BIAS_DATA});
    iomap["B_gemm2"] = B_gemm2;
  }

  shared_ptr<Tensor<T>> getTensor() {
    return std::get<std::shared_ptr<Tensor<T>>>(iomap["output"]);
  }

  void forward() {
    // Convolution 1
    ConvNode conv1("input", "W1", "conv1_output", {1, 1}, {2, 2, 2, 2}, {5, 5}, {1, 1}, std::nullopt, 1);
    conv1.forward(iomap);

    // ReLU
    ReLUNode relu1("conv1_output", "relu1_output");
    relu1.forward(iomap);
    auto conv1_output_tensor = std::get<std::shared_ptr<Tensor<T>>>(iomap["relu1_output"]);
    auto refrence_tensor = tensor_mml_p<double>({CONV1_OUTPUT_SHAPE}, {CONV1_OUTPUT_DATA});
    // Compare the output tensor with the expected output tensor.
    //std::cout << "conv1_output_tensor data: " << *conv1_output_tensor << std::endl;
    //tensors_are_close(*conv1_output_tensor, *refrence_tensor, 0.07);

    // Max Pooling
    MaxPoolingNode_mml maxpool1 = MaxPoolingNode_mml("relu1_output",
                                                     vector<string>{"maxpool1_output", "maxpool1_indices"},
                                                     array_mml({2UL, 2UL}), array_mml({2UL, 2UL}), "VALID", 0UL,
                                                     array_mml({1UL, 1UL}), array_mml({0UL, 0UL, 0UL, 0UL}), 0UL);
    maxpool1.forward(iomap);

    // Convolution 2
    ConvNode conv2("maxpool1_output", "W2", "conv2_output", {1, 1}, {0, 0, 0, 0}, {5, 5}, {1, 1}, std::nullopt, 1);
    conv2.forward(iomap);

    // ReLU
    ReLUNode relu2("conv2_output", "relu2_output");
    relu2.forward(iomap);

    // Max Pooling
    MaxPoolingNode_mml maxpool2 = MaxPoolingNode_mml("relu2_output",
                                                     vector<string>{"maxpool2_output", "maxpool2_indices"},
                                                     array_mml({2UL, 2UL}), array_mml({2UL, 2UL}), "VALID", 0UL,
                                                     array_mml({1UL, 1UL}), array_mml({0UL, 0UL, 0UL, 0UL}), 0UL);

    // MaxPoolingNode_mml maxpool2("relu2_output", {"maxpool2_output"}, {2, 2}, {2, 2}, "VALID");
    maxpool2.forward(iomap);

    // Convolution 3
    ConvNode conv3("maxpool2_output", "W3", "conv3_output", {1, 1}, {0, 0, 0, 0}, {5, 5}, {1, 1}, std::nullopt, 1);
    conv3.forward(iomap);

    // ReLU
    ReLUNode relu3("conv3_output", "relu3_output");
    relu3.forward(iomap);

    // Reshape
    auto batch_size = std::get<std::shared_ptr<Tensor<T>>>(iomap["relu3_output"])->get_shape()[0];
    auto shape_tensor = make_shared<Tensor_mml<int64_t>>(array_mml<uli>{2}, array_mml<int64_t>{static_cast<int64_t>(batch_size), -1});
    iomap["reshape_shape"] = shape_tensor;

    reshapeNode reshape1("relu3_output", "reshape_shape", "reshape1_output");
    reshape1.forward(iomap);

    // Gemm 1
    GemmNode gemm1("reshape1_output", "W_gemm1", "gemm1_output", "B_gemm1", 1.0f, 1.0f, 0, 1);
    gemm1.forward(iomap);

    // ReLU
    ReLUNode relu4("gemm1_output", "relu4_output");
    relu4.forward(iomap);

    // Gemm 2
    GemmNode gemm2("relu4_output", "W_gemm2", "gemm2_output", "B_gemm2", 1.0f, 1.0f, 0, 1);
    gemm2.forward(iomap);

    // LogSoftMax
    LogSoftMaxNode logSoftMax("gemm2_output", "output");
    logSoftMax.forward(iomap);
  }
};

/**
 * @brief Test the forward pass of the LeNet model.
 * Later on this will make use of results from Pytorch and compare them.
 * I just want to make sure that the forward pass is working first.
 */
TEST(test_LeNet, test_LeNet_forward) {
  auto TensorToProcess = tensor_mml_p<double>({INPUT_TENSOR_SHAPE}, {INPUT_TENSOR_DATA});
  LeNetModel<double> model(TensorToProcess, true);

  model.forward();
  auto output_tensor = model.getTensor();

  auto expected_output = tensor_mml_p<double>({OUTPUT_TENSOR_SHAPE}, {OUTPUT_TENSOR_DATA});
  std::cout << "Output tensor: " << *output_tensor << std::endl;

  ASSERT_TRUE(tensors_are_close(*output_tensor, *expected_output, 0.05));

  int max_index = Arithmetic_mml<double>().arg_max(output_tensor);
  ASSERT_TRUE(max_index == PREDICTED_CLASS);  // Compare against the predicted class.
}
