#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

/**
 * @class LeakyReLUNode
 * @brief A class representing a LeakyReLU node in a computational graph.
 *
 * This class inherits from the Node class and represents the rectified linear
 * function (LeakyReLU) node in a computational graph. It performs the forward
 * pass computation applying ReLU elementwise.
 */
class LeakyReLUNode : public Node {
public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for LeakyReLUNode.
   *
   * @param X Unique string key to the input tensor.
   * @param Y Unique string key to the output tensor.
   * @param alpha Coefficient of leakage. Default = 0.01
   */
  LeakyReLUNode(std::string X, std::string Y, float alpha = 0.01f);

  /**
   * @brief Constructor for LeakyReLUNode from JSON.
   * 
   * @param node JSON object representing the LeakyReLU node.
   */
  LeakyReLUNode(const json &node);

  /**
   * @brief Perform the forward pass computation using LeakyReLUNode activation
   * function.
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
  ///@brief Pointer to input tensor
  std::string X;
  ///@brief Pointer to output tensor
  std::string Y;
  ///@brief Coefficient of leakage
  float alpha;
};
