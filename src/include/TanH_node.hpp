#pragma once

#include <cmath>
#include <stdexcept>
#include <variant>

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "mml_tensor.hpp"

/**
 * @class TanHNode
 * @brief A class representing a TanH node in a computational graph.
 *
 * This class inherits from the Node class and represents the TanH
 * activation in a computational graph. It performs the forward
 * pass computation applying tanh elementwise.
 */
template <typename T>
class TanHNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
          std::is_same_v<T, double>,
      "TanHNode supports only float, double, int32_t, int64_t");

 public:
  // Type constraints: no bfloat16 or float16 for now (not native to C++17).
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for TanHNode.
   *
   * @param X Shared pointer to the input tensor X.
   * @param Y Shared pointer to the output tensor Y.
   */
  TanHNode(std::shared_ptr<AbstractTensor> X,
           std::shared_ptr<AbstractTensor> Y)
      : X(X), Y(Y) {}

  /**
   * @brief Perform the forward pass computation applying tanh.
   */
  void forward() override {
    if (!areInputsFilled())
      throw std::runtime_error("TanHNode inputs are not fully set.");

    if (!X)
      throw std::runtime_error("Failed to cast X to Tensor_mml<T>.");

    if (!Y)
      throw std::runtime_error("Output tensor Y is not allocated.");

    Arithmetic_mml<T> arithmetic;
    arithmetic.elementwise_in_place(X, [](T x) { return std::tanh(x); });
    Y->update_from(*X);
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

    auto x_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(X);
    auto valueX_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueX);
    if (!x_mml || !valueX_mml)
      throw std::runtime_error("Failed to cast X or input X to Tensor_mml<T>.");
    x_mml->update_from(*valueX_mml);
  }

  /**
   * @brief Check if the output(s) are filled.
   *
   * @return True if the output(s) are filled, false otherwise.
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
  // Input
  std::shared_ptr<AbstractTensor> X;  // Input tensor X.

  // Output
  std::shared_ptr<AbstractTensor> Y;  // Output tensor Y.
};