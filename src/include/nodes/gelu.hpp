#pragma once

#include <string>
#include <variant>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"


/**
 * @class GeluNode
 * @brief A class representing a Gelu (Gaussian Error Linear Units) node in a
 * computational graph.
 *
 * This class inherits from the Node class and represents the gaussian error
 * linear units std::function in a computational graph. The std::function is
 * applied elementwise.
 */
class GeluNode : public Node {
 public:
  using T = std::variant<double, float>;

  /**
   * @brief Constructor for GeluNode.
   *
   * @param X Unique std::string key to the tensor X.
   * @param Y Unique std::string key to the output tensor.
   * @param approximate Gelu approximation algorithm. Accepts 'std::tanh' and
   * 'none'. Default = 'none'.
   */
  GeluNode(std::string X, std::string Y, std::string approximate = "none");

  /**
   * @brief Constructor for GeluNode from JSON.
   *
   * @param node JSON object representing the Gelu node.
   */
  explicit GeluNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation using Gelu activation
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
  ///@brief Pointer to the input tensor
  std::string X;
  ///@brief Pointer to output tensor
  std::string Y;
  ///@brief Gelu approximation algorithm
  std::string approximate;
};