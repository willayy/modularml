#pragma once

#include <optional>
#include <string>
#include <variant>

#include "nodes/a_node.hpp"
#include "nlohmann/json_fwd.hpp"

/**
 * @class DropoutNode
 * @brief A class representing a Dropout node in a computational graph.
 *
 * This class inherits from the Node class and represents the Dropout node
 * in a computational graph. It performs dropout only if training is set to
 * True.
 */
class DropoutNode : public Node {
public:
  using T = std::variant<double, float>;
  using T2 = std::variant<bool>;

  /**
   * @brief Constructor for DroputNode.
   *
   * @param data Shared pointer to the input tensor data.
   * @param output Shared pointer to the output tensor output.
   * @param mask Optional shared pointer to the output tensor mask.
   * @param ratio Dropout ratio, 0.5 by default.
   * @param training_mode Training mode, False by default.
   * @param seed Random seed, None by default.
   */
  DropoutNode(std::string data, std::string output,
              std::optional<std::string> mask = std::nullopt, float ratio = 0.5,
              bool training_mode = false,
              std::optional<int> seed = std::nullopt);

  /**
   * @brief Constructor for DropoutNode from JSON.
   *
   * @param node JSON object representing the Dropout node.
   */
  explicit DropoutNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass using dropout.
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
  // Inputs
  std::string data; // Input tensor.

  // Outputs
  std::string output;              // Output tensor.
  std::optional<std::string> mask; // Optional output tensor mask.

  // Attributes
  float ratio;             // Dropout ratio.
  bool training_mode;      // Training mode.
  std::optional<int> seed; // Random seed.
};
