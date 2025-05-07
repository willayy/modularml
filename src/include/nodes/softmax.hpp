#pragma once

#include <stddef.h>

#include <string>
#include <variant>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class SoftMaxNode
 * @brief A class representing a SoftMax node in a computational graph.
 *
 * This class inherits from the Node class and represents the SoftMax node
 * in a computational graph. It performs the forward pass computation applying
 * SoftMax along the specified axis.
 */
class SoftMaxNode : public Node {
 public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for SoftMaxNode.
   *
   * @param X Shared pointer to the tensor X.
   * @param Y Shared pointer to the output tensor.
   * @param axis Integer representing along which axis SoftMax is applied to.
   * (default -1)
   */
  SoftMaxNode(const std::string &X, const std::string &Y, size_t axis = -1);

  /**
   * @brief Constructor for SoftMaxNode from JSON.
   *
   * @param node JSON object representing the SoftMax node.
   */
  explicit SoftMaxNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation using SoftMax activation
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
  std::string X;  // Input tensor X.
  std::string Y;  // Output tensor Y.
  size_t axis;
};
