#pragma once

#include "nodes/a_node.hpp"

class MaxPoolNode : public Node {
 public:
  using T = std::variant<float, double, int8_t, uint8_t>;

  /**
   * @brief Constructor for MaxPoolNode.
   *
   * @param X Input tensor name.
   * @param Y Output tensor name.
   * @param indices Optional indices tensor name (default: nullopt).
   * @param auto_pad Padding type (default: "NOTSET").
   * @param ceil_mode Ceil mode (default: 0).
   * @param kernel_shape Kernel shape.
   * @param pads Padding values.
   * @param storage_order Storage order (default: 0).
   * @param strides Stride values.
   */
  MaxPoolNode(const std::string &X, const std::string &Y, const std::vector<int> &kernel_shape,
              const std::optional<std::string> &indices = std::nullopt,
              const std::string &auto_pad = "NOTSET", int ceil_mode = 0,
              const std::vector<int> &dilations = {}, const std::vector<int> &pads = {},
              int storage_order = 0, const std::vector<int> &strides = {});

  /**
   * @brief Constructor for MaxPoolNode.
   *
   * @param node JSON object representing the MaxPool node.
   */
  explicit MaxPoolNode(const nlohmann::json& node);

  /**
   * @brief Perform the forward pass computation of MaxPool.
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
  std::optional<std::string> indices;

  // Attributes
  std::string auto_pad;
  int ceil_mode;
  std::vector<int> dilations;
  std::vector<int> kernel_shape;
  std::vector<int> pads;
  int storage_order;
  std::vector<int> strides;
};