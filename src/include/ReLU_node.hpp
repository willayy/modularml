#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "mml_tensor.hpp"

/**
 * @class ReLUNode
 * @brief A class representing a ReLU node in a computational graph.
 *
 * This class inherits from the Node class and represents the rectified linear function (ReLU) node
 * in a computational graph. It performs the forward pass computation applying ReLU elementwise.
 */
template <typename T>
class ReLUNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
      std::is_same_v<T, double> ||
      std::is_same_v<T, int32_t> ||
      std::is_same_v<T, int64_t>,
      "ReLUNode supports only float, double, int32_t, int64_t");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for ReLUNode.
   *
   * @param X Shared pointer to the tensor X.
   * @param Y Shared pointer to the output tensor.
   */
  ReLUNode(std::shared_ptr<AbstractTensor> X, std::shared_ptr<AbstractTensor> Y);

  /**
   * @brief Perform the forward pass computation using ReLU activation function.
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
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

 private:
  std::shared_ptr<AbstractTensor> X;  // Input tensor X.
  std::shared_ptr<AbstractTensor> Y;  // Output tensor Y.
};

#include "../ReLU_node.tpp"