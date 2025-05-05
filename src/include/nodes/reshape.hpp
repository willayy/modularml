#pragma once

#include <stdint.h>

#include <string>
#include <variant>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class reshapeNode
 * @brief A node that reshapes a tensor to a new shape while preserving its
 * data.
 *
 * The reshape operation changes the dimensions of a tensor without changing its
 * data. The total number of elements in the tensor remains the same, but the
 * arrangement of elements is changed according to the specified shape. This
 * operation is widely used in neural networks for transforming data between
 * layers with different dimensional requirements.
 */
class reshapeNode : public Node {
 public:
  /**
   * @brief Type alias for supported data types in reshape operations
   */
  using T = std::variant<double, float, int16_t, int32_t, int64_t, int8_t,
                         uint16_t, uint32_t, uint64_t, uint8_t>;

  /**
   * @brief Type alias for shape tensor data types
   */
  using ShapeDataType = std::variant<int64_t>;

  /**
   * @brief Constructor for reshapeNode with explicit tensor names and
   * parameters.
   *
   * @param data Name of the input tensor to be reshaped
   * @param shape Name of the tensor containing the target shape dimensions
   * @param reshaped Name of the output tensor that will contain the reshaped
   * data
   * @param allowzero When set to 1, allows dimensions with value 0 in the shape
   * tensor to be preserved from the input tensor's shape (default: 0)
   */
  reshapeNode(const std::string &data, const std::string &shape,
              const std::string &reshaped, int allowzero = 0);

  /**
   * @brief Constructor for reshapeNode from JSON representation.
   *
   * This constructor parses the JSON definition from an ONNX or similar model
   * format to extract the tensor names and parameters for the reshape
   * operation.
   *
   * @param node JSON object representing the reshape node definition
   */
  explicit reshapeNode(const nlohmann::json &node);

  /**
   * @brief Performs the forward pass of the reshape operation.
   *
   * This method reshapes the input tensor according to the dimensions specified
   * in the shape tensor. The reshaping process follows these steps:
   * 1. Verify inputs are available in the iomap
   * 2. Extract shape information from the shape tensor
   * 3. Handle special case of dimension value -1 (automatically infer this
   * dimension)
   * 4. Create a new tensor with the same data but reshaped dimensions
   * 5. Store the result in the output tensor
   *
   * The total number of elements in the output tensor must match the input
   * tensor. A dimension value of -1 indicates that this dimension should be
   * automatically calculated based on the total number of elements and other
   * dimensions.
   *
   * @param iomap Map containing input and output tensors indexed by name
   * @throws std::runtime_error If inputs are invalid, dimensions are
   * incompatible, or multiple -1 values are present in the shape
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  /**
   * @brief Gets the names of input tensors required by this node.
   *
   * @return A vector containing the names of the input tensors (data and shape)
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Gets the name of the output tensor produced by this node.
   *
   * @return A vector containing the output tensor name (reshaped)
   */
  std::vector<std::string> getOutputs() override;

 private:
  /**
   * @brief Name of the input tensor containing the data to be reshaped
   */
  std::string data;

  /**
   * @brief Name of the input tensor containing the target shape dimensions
   */
  std::string shape;

  /**
   * @brief Name of the output tensor that will contain the reshaped data
   */
  std::string reshaped;

  /**
   * @brief Controls handling of zero values in the shape tensor
   *
   * When set to 1, dimensions with value 0 in the shape tensor are
   * replaced with the corresponding dimension from the input tensor's shape.
   */
  int allowzero;
};
