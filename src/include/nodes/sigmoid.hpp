#pragma once

#include <string>
#include <variant>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class SigmoidNode
 * @brief Node that implements the sigmoid activation function.
 *
 * The sigmoid function is defined as f(x) = 1 / (1 + exp(-x)) and maps any
 * input to a value between 0 and 1. It's commonly used as an activation
 * function in neural networks, particularly in the output layer for binary
 * classification.
 *
 * This node applies the sigmoid function element-wise to the input tensor.
 */
class SigmoidNode : public Node {
 public:
  /**
   * @brief Type alias for supported floating-point types
   */
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for SigmoidNode with explicit tensor names.
   *
   * @param X Name of the input tensor
   * @param Y Name of the output tensor that will store the result
   */
  SigmoidNode(const std::string &X, const std::string &Y);

  /**
   * @brief Constructor for SigmoidNode from JSON representation.
   *
   * This constructor parses the JSON definition from an ONNX or similar model
   * format to extract the tensor names for the sigmoid operation.
   *
   * @param node JSON object representing the Sigmoid node definition
   */
  explicit SigmoidNode(const nlohmann::json &node);

  /**
   * @brief Performs the forward pass computation of the sigmoid function.
   *
   * This method applies the sigmoid function f(x) = 1 / (1 + exp(-x))
   * element-wise to the input tensor and stores the result in the output
   * tensor.
   *
   * @param iomap Map containing input and output tensors indexed by name
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  /**
   * @brief Gets the name of the input tensor required by this node.
   *
   * @return A vector containing the input tensor name
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Gets the name of the output tensor produced by this node.
   *
   * @return A vector containing the output tensor name
   */
  std::vector<std::string> getOutputs() override;

 private:
  /**
   * @brief Name of the input tensor
   */
  std::string X;

  /**
   * @brief Name of the output tensor
   */
  std::string Y;
};
