#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_tensor.hpp"

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
template <typename T> class FlattenNode : public Node {
public:
  using AbstractTensor = Tensor<T>;

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
  FlattenNode(shared_ptr<AbstractTensor> X, shared_ptr<AbstractTensor> Y,
              uli axis = 1);

  /**
   * @brief Performs the flattening operation on the input tensor.
   *
   * Transforms the input tensor into a 2D tensor along the specified axis
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   *
   * @return True if the input(s) are filled, false otherwise.
   */
  bool areInputsFilled() const override;

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set.
   */
  void setInputs(const array_mml<GeneralDataTypes> &inputs) override;

  /**
   * @brief Check if the output(s) are filled.
   *
   * @return True if the output(s) are filled, false otherwise.
   */
  bool areOutputsFilled() const override;

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

private:
  /**
   * @brief Input data tensor for the node.
   *
   * The input tensor can have any shape and any type.
   */
  shared_ptr<Tensor<T>> X;

  /**
   * @brief Output data tensor for the node.
   *
   * Contains the result after the forward pass, the shape of the tensor will
   * always be 2D.
   */
  shared_ptr<Tensor<T>> Y;

  /**
   * @brief The axis along which the flattening operation is performed.
   *
   * Allows only non-negative values, default is axis=1.
   */
  uli axis;

  uli get_axis() const;
};

#include "../flatten_node.tpp"