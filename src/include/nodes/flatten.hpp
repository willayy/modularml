#pragma once

#include <stdint.h>

#include <string>
#include <variant>

#include "a_node.hpp"
#include "nlohmann/json_fwd.hpp"

/**
 * @class FlattenNode
 * @brief A node that flattens an input tensor along a specified axis.
 * Implements the Node interface.
 *
 * The FlattenNode reshapes the input tensor into a 2D tensor (matrix),
 * where dimensions before the `axis` are collapsed into the first dimension
 * and dimensions after the `axis` are collapsed into the second dimension.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
class FlattenNode : public Node {
 public:
  using T = std::variant<double, float, int16_t, int32_t, int64_t, int8_t,
                         uint16_t, uint32_t, uint64_t, uint8_t>;

  /**
   * @brief Constructor for FlattenNode.
   *
   * Initializes a FlattenNode that flattens the input tensor `X` and stores
   * the result in the output tensor `Y`. The flattening is performed along
   * the specified `axis`.
   *
   * @param X A shared pointer to the input tensor.
   * @param Y A shared pointer to the output tensor, which will hold the
   * flattened result.
   * @param axis The axis along which the flattening operation is performed.
   *             Defaults to 1 (collapsing all dimensions before this axis into
   * the first dimension).
   */
  FlattenNode(std::string X, std::string Y, int axis = 1);

  /**
   * @brief Constructor for FlattenNode from JSON.
   *
   * @param node JSON object representing the Flatten node.
   */
  FlattenNode(const nlohmann::json &node);

  /**
   * @brief Performs the flattening operation on the input tensor.
   *
   * Transforms the input tensor into a 2D tensor along the specified axis
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
  /**
   * @brief Input data tensor for the node.
   *
   * The input tensor can have any shape and any type.
   */
  std::string X;

  /**
   * @brief Output data tensor for the node.
   *
   * Contains the result after the forward pass, the shape of the tensor will
   * always be 2D.
   */
  std::string Y;

  /**
   * @brief The axis along which the flattening operation is performed.
   *
   * Allows only non-negative values, default is axis=1.
   */
  int axis;

  int get_axis() const;
};
