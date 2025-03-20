#pragma once

#include "a_node.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"

/**
 * @class reshapeNode
 * @brief A class representing a reshape node in a computational graph.
 *
 * This class inherits from the Node class and represents the reshape node
 * in a computational graph. It performs reshape after being given an
 * input tensor and a shape tensor.
 */
template <typename T>
class reshapeNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
          std::is_same_v<T, double> ||
          std::is_same_v<T, int32_t> ||
          std::is_same_v<T, int64_t> ||
          std::is_same_v<T, bool> ||
          std::is_same_v<T, string>,
      "reshapeNode supports only float, double, int32_t, int64_t, bool, string");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for reshapeNode.
   *
   * @param data Shared pointer to the tensor data.
   * @param shape Shared pointer to the shape tensor (int64_t).
   * @param reshaped Shared pointer to the reshaped tensor (output).
   * @param allowzero =0 by default. allowzero=1 indicates that if any value in the ‘shape’ input is set to zero, the zero value is honored
   */
  reshapeNode(shared_ptr<const AbstractTensor> data, shared_ptr<const Tensor<int64_t>> shape,
              shared_ptr<AbstractTensor> reshaped, int allowzero = 0);

  /**
   * @brief Performs the forward pass of the reshape operation.
   *
   * This function reshapes the input tensor to the specified shape.
   * It checks if the inputs are filled, casts the data and shape to the appropriate tensor types,
   * and allocates the output tensor if necessary. It then extracts the shape values from the shape tensor,
   * handles the special case where a dimension is set to -1 (inferred dimension), and computes the new shape.
   * Finally, it copies the data from the input tensor to the reshaped output tensor and applies the reshape.
   *
   * @throws std::runtime_error if the inputs are not fully set, if the data or shape cannot be cast to the appropriate tensor types,
   *                            if the output tensor is not allocated, if multiple -1 values are present in the shape tensor,
   *                            or if the inferred dimension does not match the total elements.
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   */
  bool areInputsFilled() const override;

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set, where inputs[0] is the data tensor and inputs[1] is the shape tensor.
   */
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override;

  /**
   * @brief Check if the output(s) are filled.
   */
  bool areOutputsFilled() const override;

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

 private:
  // tensors
  shared_ptr<const AbstractTensor> data;    // Input tensor data.
  shared_ptr<const Tensor<int64_t>> shape;  // Input tensor shape.
  shared_ptr<AbstractTensor> reshaped;      // Output tensor reshaped.

  // attributes
  int allowzero;
};

#include "../reshape_node.tpp"