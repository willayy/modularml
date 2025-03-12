#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "a_tensor.hpp"

/**
 * @class SwishNode
 * @brief A class representing a Swish node in a computational graph.
 *
 * This class inherits from the Node class and represents the Swish
 * activation in a computational graph. It performs the forward
 * pass computation applying swish elementwise.
 */
template <typename T>
class SwishNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
      std::is_same_v<T, double>,
      "SwishNode supports only float, double");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for SwishNode.
   *
   * @param X Shared pointer to the input tensor X.
   * @param Y Shared pointer to the output tensor Y.
   */
  SwishNode(std::shared_ptr<AbstractTensor> X,
            std::shared_ptr<AbstractTensor> Y);

  /**
   * @brief Perform the forward pass computation applying swish.
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
   * @param inputs The input data to be set, where X is inputs[0].
   */
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override;

  /**
   * @brief Check if the output(s) are filled.
   *
   * @return True if the output(s) are filled, false otherwise.
   */
  bool areOutputsFilled() const override;

  /**
   * @brief Get the output(s) of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

 private:
  // Input
  shared_ptr<const AbstractTensor> X;  // Input tensor X.

  // Output
  std::shared_ptr<AbstractTensor> Y;  // Output tensor Y.
};

#include "../Swish_node.tpp"
