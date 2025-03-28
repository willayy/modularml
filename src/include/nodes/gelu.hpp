#pragma once

#include "nodes/a_node.hpp"

/**
 * @class GeluNode
 * @brief A class representing a Gelu (Gaussian Error Linear Units) node in a
 * computational graph.
 *
 * This class inherits from the Node class and represents the gaussian error
 * linear units function in a computational graph. The function is applied
 * elementwise.
 */
class GeluNode : public Node {
public:
  using T = std::variant<double, float>;

  /**
   * @brief Constructor for GeluNode.
   *
   * @param X Unique string key to the tensor X.
   * @param Y Unique string key to the output tensor.
   * @param approximate Gelu approximation algorithm. Accepts 'tanh' and 'none'.
   * Default = 'none'.
   */
  GeluNode(std::string X, std::string Y,
           string approximate = "none");

  /**
   * @brief Constructor for GeluNode from JSON.
   * 
   * @param node JSON object representing the Gelu node.
   */
  GeluNode(const json &node);

  /**
   * @brief Perform the forward pass computation using Gelu activation function.
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
  ///@brief Pointer to the input tensor
  std::string X;
  ///@brief Pointer to output tensor
  std::string Y;
  ///@brief Gelu approximation algorithm
  string approximate;
};