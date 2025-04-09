#pragma once

#include "nodes/a_node.hpp"

/**
 * @class TanHNode
 * @brief A class representing a TanH node in a computational graph.
 *
 * This class inherits from the Node class and represents the TanH
 * activation in a computational graph. It performs the forward
 * pass computation applying std::tanh elementwise.
 */
class TanHNode : public Node {
public:
  using T = std::variant<double, float>;

  /**
   * @brief Constructor for TanHNode.
   *
   * @param X Shared pointer to the input tensor X.
   * @param Y Shared pointer to the output tensor Y.
   */
  TanHNode(std::string X, std::string Y);

  /**
   * @brief Constructor for TanHNode from JSON.
   *
   * @param node JSON object representing the TanH node.
   */
  TanHNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation applying std::tanh.
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
  // Input
  std::string X; // Input tensor X.

  // Output
  std::string Y; // Output tensor Y.
};
