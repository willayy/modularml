#pragma once

#include "nodes/a_node.hpp"

/**
 * @class Constant Node
 * @brief A class representing a Constant node in a computational graph.
 */
class ConstantNode : public Node {
public:
  /**
   * @brief Constructor for ConstantNode.
   *
   * @param output Unique std::string key to the input tensor.
   * @param value Tensor of std::variant GeneralDataTypes that will be assigned
   * to output.
   */
  ConstantNode(std::string output, GeneralDataTypes value);

  /**
   * @brief Constructor for ConstantNode from JSON.
   *
   * @param node JSON object representing the Constant node.
   */
  ConstantNode(const nlohmann::json &node);

  /**§
   * @brief Perform the forward pass.
   */
  void
  forward(std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

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
  ///@brief Unique std::string key to output tensor
  std::string output;
  ///@brief Value of the constant
  GeneralDataTypes value;

  // Should support more constant value attributes
};
