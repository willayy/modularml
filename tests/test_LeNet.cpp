#include <gtest/gtest.h>

#include <modularml>

template <typename T>
class LeNetModel {
 private:
  shared_ptr<Tensor<T>> input_tensor;
  shared_ptr<Tensor<T>> output_tensor;

  shared_ptr<Tensor_mml<float>> W1, w2, W3, W_gemm1, W_gemm2, B_gemm1, B_gemm2;
  shared_ptr<Tensor_mml<T>> conv_output_tensor1, conv_output_tensor2a, conv_output_tensor2b, conv_output_tensor3;
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
    conv_output_tensor2a = make_shared<Tensor_mml<T>>(conv_shape);
    conv_output_tensor2b = make_shared<Tensor_mml<T>>(conv_shape);
    conv_output_tensor3 = make_shared<Tensor_mml<T>>(conv_shape);

    // Output tensor construtors:
    // Allocate minimal output tensors (single-element tensors)
    output_gemm1 = make_shared<Tensor_mml<T>>(array_mml<int>{1});
    output_gemm2 = make_shared<Tensor_mml<T>>(array_mml<int>{1});
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

    w2 = std::make_shared<Tensor_mml<T>>(array_mml<int>{16, 6, 5, 5});
    kaimingUniform(w2, 6, 5, gen);

    W3 = std::make_shared<Tensor_mml<T>>(array_mml<int>{120, 16, 5, 5});
    kaimingUniform(W3, 16, 5, gen);

    W_gemm1 = std::make_shared<Tensor_mml<T>>(array_mml<int>{84, 480});
    kaimingUniform(W_gemm1, 120, 480, gen);

    W_gemm2 = std::make_shared<Tensor_mml<T>>(array_mml<int>{10, 84});
    kaimingUniform(W_gemm2, 84, 84, gen);

    // Bias for Gemm
    B_gemm1 = std::make_shared<Tensor_mml<T>>(array_mml<int>{84});
    B_gemm1->fill(0.0f);

    B_gemm2 = std::make_shared<Tensor_mml<T>>(array_mml<int>{10});
    B_gemm2->fill(0.0f);

    // Convolutional layers
    conv1 = make_unique<ConvNode<T>>(input_tensor, W1, conv_output_tensor1,
                                     array_mml<int>{1, 1},        // dilation = 1
                                     array_mml<int>{2, 2, 2, 2},  // padding = 2
                                     array_mml<int>{5, 5},        // kernel size = 5
                                     array_mml<int>{1, 1},        // stride = 1
                                     std::nullopt,                // No bias
                                     1);                          // groups = 1

    conv2 = make_unique<ConvNode<T>>(input_tensor, w2, conv_output_tensor2a,
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
        W_gemm1,       // B (Weight tensor of shape (84, 480))
        output_gemm1,  // Y (Output tensor of shape (batch_size, 84))
        B_gemm1,       // C (Bias tensor of shape (84,))
        1.0f,          // alpha (default 1.0)
        1.0f,          // beta (default 1.0)
        0,             // transA = 0 (No transpose for input)
        1              // transB = 1 (Transpose weight matrix)
    );

    gemm2 = make_unique<GemmNode<T>>(
        input_tensor,  // A (Input tensor of shape (batch_size, 84))
        W_gemm2,       // B (Weight tensor of shape (10, 84))
        output_gemm2,  // Y (Output tensor of shape (batch_size, 10))
        B_gemm2,       // C (Bias tensor of shape (10,))
        1.0f,          // alpha (default 1.0)
        1.0f,          // beta (default 1.0)
        0,             // transA = 0 (No transpose for input)
        1              // transB = 1 (Transpose weight matrix)
    );

    // Other layers
    reluNode = make_unique<ReLUNode<T>>(input_tensor, output_tensor_ReLU);
    maxPoolNode = make_unique<MaxPoolingNode_mml<T>>(vector<int>{2, 2}, vector<int>{2, 2}, std::static_pointer_cast<Tensor_mml<T>>(input_tensor), "VALID");
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

    // Max Pooling
    maxPoolNode->forward();
    *output_tensor_MaxPool = *std::get<std::shared_ptr<Tensor<T>>>(maxPoolNode->getOutputs()[0]);

    // Print the shape of output_tensor_MaxPool
    std::cout << "Shape of output_tensor_MaxPool: " << output_tensor_MaxPool->get_shape() << std::endl;

    // Print the shape of W2
    std::cout << "Shape of W2: " << w2->get_shape() << std::endl;

    *input_tensor = *output_tensor_MaxPool;

    // Convolution 2
    conv2->forward();
    *input_tensor = *conv_output_tensor2a;

    // Relu
    reluNode->forward();
    *input_tensor = *output_tensor_ReLU;

    // Max Pooling
    maxPoolNode->forward();
    *input_tensor = *output_tensor_MaxPool;

    // Add
    // copy the tensor
    auto copytensor = make_shared<Tensor_mml<T>>(*output_tensor_MaxPool);
    // Update the inputs
    array_mml<GeneralDataTypes> inputs({copytensor, input_tensor});
    addNode->setInputs(inputs);
    // Add
    addNode->forward();
    *input_tensor = *output_tensor_add;

    // Convolution 3
    conv3->forward();
    *input_tensor = *conv_output_tensor3;

    // Relu
    reluNode->forward();
    *input_tensor = *output_tensor_ReLU;

    // Flatten
    auto batch_size = input_tensor->get_shape()[0];
    auto shape_tensor = make_shared<Tensor_mml<int64_t>>(array_mml<int>{batch_size, -1});
    reshapeNode1 = make_unique<reshapeNode<T>>(input_tensor, shape_tensor, output_tensor_Reshape);
    reshapeNode1->forward();
    *input_tensor = *output_tensor_Reshape;

    // Gemm 1
    gemm1->forward();
    *input_tensor = *output_gemm1;

    // Relu
    reluNode->forward();
    *input_tensor = *output_tensor_ReLU;

    // Gemm 2
    gemm2->forward();
    *input_tensor = *output_gemm2;

    // LogSoftMax
    logSoftMaxNode->forward();

    // FINAL RESULT
    *output_tensor = *output_tensor_logSoftMax;
  }

  ~LeNetModel() {
    input_tensor.reset();
    output_tensor.reset();
  }
};

TEST(test_LeNet, test_LeNet_forward) {
  array_mml<int> tensor_shape = array_mml<int>({1, 1, 32, 32});
  array_mml<float> tensor_data = array_mml<float>(vector<float>(32 * 32, 1.0f));

  auto tensor = make_shared<Tensor_mml<float>>(tensor_shape, tensor_data);
  LeNetModel<float> model(tensor);

  model.forward();
  auto output_tensor = model.getTensor();

  array_mml<int> expected_shape = array_mml<int>({1, 1, 14, 14});
  array_mml<float> expected_data = array_mml<float>(vector<float>(14 * 14, 1.0f));
  auto expected_output = make_shared<Tensor_mml<float>>(expected_shape, expected_data);

  ASSERT_EQ(*output_tensor, *expected_output);
}
