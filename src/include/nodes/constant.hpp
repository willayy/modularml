#pragma once

#include <string>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class ConstantNode
 * @brief A node that outputs a constant tensor value in a computational graph.
 *
 * The ConstantNode represents a constant value in the computational graph. It
 * has no inputs and produces a single output tensor with a predefined value
 * that does not change during inference.
 */
class ConstantNode : public Node {
 public:
  /**
   * @brief Constructor for ConstantNode with explicit value.
   *
   * @param output The name of the output tensor that will contain the constant
   * value
   * @param value The constant tensor value to be output by this node
   */
  ConstantNode(const std::string &output, GeneralDataTypes value);

  /**
   * @brief Constructor for ConstantNode from JSON representation.
   *
   * This constructor parses the JSON definition from an ONNX or similar model
   * format to extract the constant value information.
   *
   * @param node JSON object representing the Constant node definition
   */
  explicit ConstantNode(const nlohmann::json &node);

  /**
   * @brief Performs the forward pass operation.
   *
   * In the case of a constant node, this simply makes the constant value
   * available in the iomap under the output tensor name.
   *
   * @param iomap Map containing input and output tensors indexed by name
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  /**
   * @brief Gets the names of input tensors required by this node.
   *
   * For a constant node, there are no inputs, so this returns an empty vector.
   *
   * @return An empty vector since constant nodes have no inputs
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Gets the name of the output tensor produced by this node.
   *
   * @return A vector containing the single output tensor name
   */
  std::vector<std::string> getOutputs() override;

 private:
  /**
   * @brief The name of the output tensor
   */
  std::string output;

  /**
   * @brief The constant value to be output
   *
   * This can be a tensor of any supported type in the GeneralDataTypes variant.
   */
  GeneralDataTypes value;
};
