#pragma once

#include <cmath>
#include <stdexcept>
#include <variant>

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
  SwishNode(shared_ptr<const AbstractTensor> X,
            shared_ptr<AbstractTensor> Y)
      : X(X), Y(Y) {}

  /**
   * @brief Perform the forward pass computation applying swish.
   */
  void forward() override {
    if (!areInputsFilled())
      throw runtime_error("SwishNode inputs are not fully set.");

    if (!X)
      throw runtime_error("Failed to cast X to Tensor_mml<T>.");

    if (!Y)
      throw runtime_error("Output tensor Y is not allocated.");

    Arithmetic_mml<T> arithmetic;
    arithmetic.elementwise(X, [](T x) {
      T sigmoid_x = static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
      return x * sigmoid_x;
    }, Y);
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
      throw runtime_error("TanHNode expects at least one input: X.");

    auto valueX = std::get<shared_ptr<AbstractTensor>>(inputs[0]);

    if (!X || !valueX)
      throw runtime_error("Failed to cast X or input X to Tensor_mml<T>.");
    
    X = std::const_pointer_cast<AbstractTensor>(valueX);
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
   * @brief Get the output(s) of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
  }

 private:
  // Input
  shared_ptr<const AbstractTensor> X;  // Input tensor X.

  // Output
  shared_ptr<AbstractTensor> Y;  // Output tensor Y.
};