#pragma once

#include <string>
#include <variant>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class AvgPoolNode
 * @brief Implementation of average pooling operation in a neural network.
 *
 * The AvgPoolNode performs average pooling, which downsamples input data by
 * taking the average value of elements in a sliding window. This is commonly
 * used in convolutional neural networks to reduce spatial dimensions and
 * provide translation invariance.
 */
class AvgPoolNode : public Node {
 public:
  /**
   * @brief Type alias for supported floating-point types
   */
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for AvgPoolNode with explicit parameters.
   *
   * @param X Input tensor name
   * @param Y Output tensor name
   * @param kernel_shape Shape of the pooling kernel/window
   * @param auto_pad Padding type (options: "NOTSET", "SAME_UPPER",
   * "SAME_LOWER", "VALID")
   * @param ceil_mode When set to 1, use ceil instead of floor to compute output
   * shape
   * @param count_include_pad Whether to include padding in the averaging
   * calculation
   * @param dilations Dilation values for the pooling window
   * @param pads Padding values for each spatial dimension (format: [x1_begin,
   * x2_begin, ..., x1_end, x2_end, ...])
   * @param strides Stride values for each spatial dimension
   */
  AvgPoolNode(const std::string &X, const std::string &Y,
              const std::vector<int> &kernel_shape,
              const std::string &auto_pad = "NOTSET", int ceil_mode = 0,
              int count_include_pad = 0, const std::vector<int> &dilations = {},
              const std::vector<int> &pads = {},
              const std::vector<int> &strides = {});

  /**
   * @brief Constructor for AvgPoolNode from JSON representation.
   *
   * This constructor parses the JSON definition from an ONNX or similar model
   * format to extract the average pooling parameters.
   *
   * @param node JSON object representing the AvgPool node
   */
  explicit AvgPoolNode(const nlohmann::json &node);

  /**
   * @brief Performs the forward pass computation of average pooling.
   *
   * Applies average pooling to the input tensor according to the specified
   * parameters and stores the result in the output tensor.
   *
   * @param iomap Map containing input and output tensors indexed by name
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  /**
   * @brief Gets the name of the input tensor required by this node.
   *
   * @return A vector containing the input tensor name
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Gets the name of the output tensor produced by this node.
   *
   * @return A vector containing the output tensor name
   */
  std::vector<std::string> getOutputs() override;

 private:
  /**
   * @brief Input tensor name
   */
  std::string X;

  /**
   * @brief Output tensor name
   */
  std::string Y;

  /**
   * @brief Padding type (options: "NOTSET", "SAME_UPPER", "SAME_LOWER",
   * "VALID")
   */
  std::string auto_pad;

  /**
   * @brief When set to 1, use ceil instead of floor to compute output shape
   */
  int ceil_mode;

  /**
   * @brief Whether to include padding in the averaging calculation
   */
  int count_include_pad;

  /**
   * @brief Dilation values for the pooling window
   */
  std::vector<int> dilations;

  /**
   * @brief Shape of the pooling kernel/window
   */
  std::vector<int> kernel_shape;

  /**
   * @brief Padding values for each spatial dimension
   *
   * Format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
   */
  std::vector<int> pads;

  /**
   * @brief Stride values for each spatial dimension
   */
  std::vector<int> strides;
};