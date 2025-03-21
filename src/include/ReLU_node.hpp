#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "a_tensor.hpp"

/**
 * @class ReLUNode
 * @brief A class representing a ReLU node in a computational graph.
 *
 * This class inherits from the Node class and represents the rectified linear function (ReLU) node
 * in a computational graph. It performs the forward pass computation applying ReLU elementwise.
 */
class ReLUNode : public Node {
 public:
  using T = std::variant<double, float, int16_t, int32_t, int64_t, int8_t>;
  /**
   * @brief Constructor for ReLUNode.
   *
   * @param X Shared pointer to the tensor X.
   * @param Y Shared pointer to the output tensor.
   */
  ReLUNode(std::string X, std::string Y);

  /**
   * @brief Constructor for ReluNode from JSON.
   *
   * @param node JSON object representing the Relu node.
   */
  ReLUNode(const json& node);

  /**
   * @brief Perform the forward pass computation using ReLU activation function.
   */
  void forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) override;

 private:
  std::string X;  // Input tensor X.
  std::string Y;  // Output tensor Y.
};

