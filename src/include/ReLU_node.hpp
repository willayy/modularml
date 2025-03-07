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
  ReLUNode(std::shared_ptr<AbstractTensor> X, std::shared_ptr<AbstractTensor> Y)
      : X(X), Y(Y) {}

  /**
   * @brief Perform the forward pass computation using ReLU activation function.
   */
  void forward() override {
    if (!areInputsFilled())
      throw std::runtime_error("ReLUNode inputs are not fully set.");

    if (!X)
      throw std::runtime_error("Failed to cast X to Tensor_mml<T>.");

    if (!Y)
      throw std::runtime_error("Output tensor Y is not allocated.");

    Arithmetic_mml<T> arithmetic;
    arithmetic.elementwise_in_place(X, [](T x) { return x > 0 ? x : 0; });
    *Y = *X;
  }

  /**
   * @brief Check if the input(s) are filled.
   */
  bool areInputsFilled() const override {
    return X && X->get_size() > 0;
  }

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set, where X is inputs[0].
   */
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override {
    if (inputs.size() < 1)
      throw std::runtime_error("TanHNode expects at least one input: X.");

    auto valueX = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);

    auto valueX_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueX);
    if (!X || !valueX_mml)
      throw std::runtime_error("Failed to cast X or input X to Tensor_mml<T>.");
    *X = *Y;
  }

  /**
   * @brief Check if the output(s) are filled.
   */
  bool areOutputsFilled() const override {
    if (!Y) return false;
    return Y->get_size() > 0;
  }

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
  }

 private:
  std::shared_ptr<AbstractTensor> X;  // Input tensor X.
  std::shared_ptr<AbstractTensor> Y;  // Output tensor Y.
};