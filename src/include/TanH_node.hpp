#pragma once

#include "a_node.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

/**
 * @class TanHNode
 * @brief A class representing a TanH node in a computational graph.
 *
 * This class inherits from the Node class and represents the TanH
 * activation in a computational graph. It performs the forward
 * pass computation applying tanh elementwise.
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
  TanHNode(const json& node);

  /**
   * @brief Perform the forward pass computation applying tanh.
   */
  void forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) override;

 private:
  // Input
  std::string X;  // Input tensor X.

  // Output
  std::string Y;  // Output tensor Y.
};
