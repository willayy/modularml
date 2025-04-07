#pragma once

#include "nodes/a_node.hpp"

/**
 * @class reshapeNode
 * @brief A class representing a reshape node in a computational graph.
 *
 * This class inherits from the Node class and represents the reshape node
 * in a computational graph. It performs reshape after being given an
 * input tensor and a shape tensor.
 */
class reshapeNode : public Node {
public:
  using T = std::variant<double, float, int16_t, int32_t, int64_t, int8_t,
                         uint16_t, uint32_t, uint64_t, uint8_t>;
  using ShapeDataType = std::variant<int64_t>;

  /**
   * @brief Constructor for reshapeNode.
   *
   * @param data Shared pointer to the tensor data.
   * @param shape Shared pointer to the shape tensor (int64_t).
   * @param reshaped Shared pointer to the reshaped tensor (output).
   * @param allowzero =0 by default. allowzero=1 indicates that if any value in
   * the ‘shape’ input is set to zero, the zero value is honored
   */
  reshapeNode(std::string data, std::string shape, std::string reshaped,
              int allowzero = 0);

  /**
   * @brief Constructor for reshapeNode from JSON.
   *
   * @param node JSON object representing the reshape node.
   */
  reshapeNode(const nlohmann::json &node);

  /**
   * @brief Performs the forward pass of the reshape operation.
   *
   * This std::function reshapes the input tensor to the specified shape.
   * It checks if the inputs are filled, casts the data and shape to the
   * appropriate tensor types, and allocates the output tensor if necessary. It
   * then extracts the shape values from the shape tensor, handles the special
   * case where a dimension is set to -1 (inferred dimension), and computes the
   * new shape. Finally, it copies the data from the input tensor to the
   * reshaped output tensor and applies the reshape.
   *
   * @throws std::runtime_error if the inputs are not fully set, if the
   * data or shape cannot be cast to the appropriate tensor types, if the output
   * tensor is not allocated, if multiple -1 values are present in the shape
   * tensor, or if the inferred dimension does not match the total elements.
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
  // tensors
  std::string data;     // Input tensor data.
  std::string shape;    // Input tensor shape.
  std::string reshaped; // Output tensor reshaped.

  // attributes
  int allowzero;
};
