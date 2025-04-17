#pragma once

#include <string>
#include <variant>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

class AvgPoolNode : public Node {
 public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for AvgPoolNode.
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
  AvgPoolNode(std::string X, std::string Y, std::vector<int> kernel_shape,
              std::string auto_pad = "NOTSET", int ceil_mode = 0,
              int count_include_pad = 0, std::vector<int> dilations = {},
              std::vector<int> pads = {}, std::vector<int> strides = {});

  /**
   * @brief Constructor for AvgPoolNode.
   *
   * @param node JSON object representing the MaxPool node.
   */
  explicit AvgPoolNode(const nlohmann::json& node);

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

  // Attributes
  std::string auto_pad;
  int ceil_mode;
  int count_include_pad;
  std::vector<int> dilations;
  std::vector<int> kernel_shape;
  std::vector<int> pads;
  std::vector<int> strides;
};