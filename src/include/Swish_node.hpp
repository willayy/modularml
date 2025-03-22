#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "a_tensor.hpp"

/**
 * @class SwishNode
 * @brief A class representing a Swish node in a computational graph.
 *
 * This class inherits from the Node class and represents the Swish
 * activation in a computational graph. It performs the forward
 * pass computation applying swish elementwise.
 */
class SwishNode : public Node {
 public:
  using T = std::variant<double, float>;
  /**
   * @brief Constructor for SwishNode.
   *
   * @param X Shared pointer to the input tensor X.
   * @param Y Shared pointer to the output tensor Y.
   */
  SwishNode(std::string X, std::string Y);

  /**
   * @brief Constructor for SwishNode from JSON.
   *
   * @param node JSON object representing the Swish node.
   */
  SwishNode(const json& node);

  /**
   * @brief Perform the forward pass computation applying swish.
   */
  void forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) override;

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
  std::string X;  // Input tensor X.

  // Output
  std::string Y;  // Output tensor Y.
};
