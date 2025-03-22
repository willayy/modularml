#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include <cmath>

/**
 * @class GeluNode
 * @brief A class representing a Gelu (Gaussian Error Linear Units) node in a
 * computational graph.
 *
 * This class inherits from the Node class and represents the gaussian error
 * linear units function in a computational graph. The function is applied
 * elementwise.
 */
template <typename T> class GeluNode : public Node {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "GeluNode supports only float, double");

public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for GeluNode.
   *
   * @param X Shared pointer to the tensor X.
   * @param Y Shared pointer to the output tensor.
   * @param approximate Gelu approximation algorithm. Accepts 'tanh' and 'none'.
   * Default = 'none'.
   */
  GeluNode(shared_ptr<const AbstractTensor> X, shared_ptr<AbstractTensor> Y,
           string approximate = "none");

  /**
   * @brief Perform the forward pass computation using Gelu activation function.
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
  ///@brief Pointer to the input tensor
  shared_ptr<const AbstractTensor> X;
  ///@brief Pointer to output tensor
  shared_ptr<AbstractTensor> Y;
  ///@brief Gelu approximation algorithm
  string approximate;
};
#include "../Gelu_node.tpp"