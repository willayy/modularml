#pragma once

#include "nodes/a_node.hpp"

/**
 * @class Sigmoid_mml
 * @brief A class that implements a tensor std::function for the Sigmoid
 * std::function.
 * @param T The data type of the tensor elements (must be float or double).
 */
class SigmoidNode : public Node {
 public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for SigmoidNode.
   * @param X Unique std::string key to the input tensor
   * @param Y Unique std::string key to the output tensor
   */
  SigmoidNode(const std::string &X, const std::string &Y);

  /**
   * @brief Constructor for SigmoidNode from JSON.
   *
   * @param node JSON object representing the Sigmoid node.
   */
  explicit SigmoidNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation using the Sigmoid
   * std::function.
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  /**
   * @brief Get inputs.
   *
   * @return The names of the inputs to the node.
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Get outputs.
   *
   * @return The names of the outputs to the node.
   */
  std::vector<std::string> getOutputs() override;

 private:
  ///@brief Unique std::string key to the input tensor
  std::string X;

  ///@brief Unique std::string key to the output tensor
  std::string Y;
};
