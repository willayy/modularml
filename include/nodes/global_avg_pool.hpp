#pragma once

#include <string>
#include <variant>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

class GlobalAvgPoolNode : public Node {
 public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for GlobalAvgPoolNode.
   *
   * @param X Input tensor name.
   * @param Y Output tensor name.
   */
  GlobalAvgPoolNode(std::string X, std::string Y);

  /**
   * @brief Constructor for GlobalAvgPoolNode.
   *
   * @param node JSON object representing the MaxPool node.
   */
  GlobalAvgPoolNode(const nlohmann::json& node);

  /**
   * @brief Perform the forward pass computation of AvgPoolNode.
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes>& iomap) override;

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
  // Inputs
  std::string X;

  // Outputs
  std::string Y;
};