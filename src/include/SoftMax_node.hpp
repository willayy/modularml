#pragma once

#include "a_node.hpp"
#include "mml_tensor.hpp"
#include "tensor_utility.hpp"
#include <memory>
#include <cmath>
#include <type_traits>
#include "mml_arithmetic.hpp"

/**
 * @brief SoftmaxNode applies the Softmax activation function along a specified axis.
 *
 * @tparam T Data type of the tensor. Must be one of float, double, int32_t, or int64_t.
 */
template <typename T>
class SoftmaxNode : public Node {
  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, double> ||
      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>,
      "SoftmaxNode supports only float, double, int32_t, int64_t");

public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for SoftmaxNode.
   *
   * @param X Shared pointer to the input tensor X.
   * @param axis Axis along which to apply softmax. Defaults to -1.
   */
  SoftmaxNode(std::shared_ptr<const AbstractTensor> X, int axis = -1);

  /**
   * @brief Executes the forward pass of the Softmax operation.
   */
  void forward() override;

  /**
   * @brief Retrieves the output tensor.
   *
   * @return The output tensor.
   */
  std::shared_ptr<AbstractTensor> getOutput();

  /**
   * @brief Overrides required virtual functions from Node.
   */
  bool areInputsFilled() const override;
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override;
  bool areOutputsFilled() const override;
  array_mml<GeneralDataTypes> getOutputs() const override;

private:
  std::shared_ptr<const AbstractTensor> input;
  std::shared_ptr<AbstractTensor> output;
  int axis;
};

#include "../SoftMax_node.tpp"