#pragma once

#include "a_node.hpp"
#include "globals.hpp"

/**
 * @class LeakyReLUNode
 * @brief A class representing a LeakyReLU node in a computational graph.
 *
 * This class inherits from the Node class and represents the rectified linear
 * function (LeakyReLU) node in a computational graph. It performs the forward
 * pass computation applying ReLU elementwise.
 */
template <typename T> class LeakyReLUNode : public Node {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "LeakyReLUNode supports only float, double");

public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for LeakyReLUNode.
   *
   * @param X Shared pointer to the input tensor.
   * @param Y Shared pointer to the output tensor.
   * @param alpha Coefficient of leakage. Default = 0.01
   */
  LeakyReLUNode(shared_ptr<const AbstractTensor> X,
                shared_ptr<AbstractTensor> Y, float alpha = 0.01f);

  /**
   * @brief Perform the forward pass computation using LeakyReLUNode activation
   * function.
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
  ///@brief Perform LeakyReLU operation
  ///@param x Value to perform LeakyReLU operation on
  T leaky_relu_operation(T x);
  ///@brief Perform LeakyReLU on each element in tensor
  void leaky_relu_elementwise();
  ///@brief Pointer to input tensor
  shared_ptr<const AbstractTensor> X;
  ///@brief Pointer to output tensor
  shared_ptr<AbstractTensor> Y;
  ///@brief Coefficient of leakage
  float alpha;
};

#include "../LeakyReLU_node.tpp"