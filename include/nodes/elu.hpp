#pragma once

#include "nodes/a_node.hpp"

/**
 * @class ELUNode
 * @brief A class that implements a tensor std::function for the ELU
 * (Exponential Linear Unit) std::function.
 */
class ELUNode : public Node {
 public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for ELUNode.
   *
   * @param X Unique std::string key to the input tensor.
   * @param Y Unique std::string key to the output tensor.
   * @param alpha Coefficient of ELU.
   */
  ELUNode(const std::string &X, const std::string &Y, float alpha = 1.0f);

  /**
   * @brief Constructor for ELUNode from JSON.
   *
   * @param node JSON object representing the ELU node.
   */
  explicit ELUNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation using the ELU std::function.
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
  ///@brief Unique std::string key to input tensor
  std::string X;
  ///@brief Unique std::string key to output tensor
  std::string Y;
  ///@brief Coefficient of ELU
  float alpha;
};
