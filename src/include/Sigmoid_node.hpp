#pragma once

#include <cmath>
#include <type_traits>

/**
 * @class Sigmoid_mml
 * @brief A class that implements a tensor function for the Sigmoid function.
 * @param T The data type of the tensor elements (must be float or double).
 */
template <typename T> class SigmoidNode : public Node {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SigmoidNode supports only float, double ");

public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for SigmoidNode.
   * @param X Pointer to the input tensor
   * @param Y Pointer to the output tensor
   */
  SigmoidNode(shared_ptr<const AbstractTensor> X, shared_ptr<AbstractTensor> Y);
  /**
   * @brief Perform the forward pass computation using the Sigmoid function.
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   * @return True if the input(s) are filled, false otherwise.
   */
  bool areInputsFilled() const override;

  /**
   * @brief Set the input(s) for the node.
   * @param inputs The input data to be set.
   */
  void setInputs(const array_mml<GeneralDataTypes> &inputs)
      override; // This function could have better
                // type safety somehow maybe.

  /**
   * @brief Check if the output(s) are filled.
   * @return True if the output(s) are filled, false otherwise.
   */
  bool areOutputsFilled() const override;

  /**
   * @brief Get the output of the node.
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

private:
  ///@brief Pointer to the input tensor
  shared_ptr<const AbstractTensor> X;

  ///@brief Pointer to the output tensor
  shared_ptr<AbstractTensor> Y;
};
#include "../Sigmoid_node.tpp"