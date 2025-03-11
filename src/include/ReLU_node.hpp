#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "a_tensor.hpp"

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
  ReLUNode(shared_ptr<const AbstractTensor> X, shared_ptr<AbstractTensor> Y)
      : X(X), Y(Y) {}

  /**
   * @brief Perform the forward pass computation using ReLU activation function.
   */
  void forward() override {
    if (!areInputsFilled())
      throw runtime_error("ReLUNode inputs are not fully set.");

    if (!X)
      throw runtime_error("Failed to cast X to Tensor<T>.");

    if (!Y)
      throw runtime_error("Output tensor Y is not allocated.");

    Arithmetic_mml<T> arithmetic;
    arithmetic.elementwise(X, [](T x) { return x > 0 ? x : 0; }, Y);
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
      throw runtime_error("ReLUNode expects at least one input: X.");

    auto valueX = std::get<shared_ptr<AbstractTensor>>(inputs[0]);

    if (!X || !valueX)
      throw runtime_error("Failed to cast X or input X to Tensor<T>.");
    
    X = std::const_pointer_cast<AbstractTensor>(valueX);
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
  shared_ptr<const AbstractTensor> X;  // Input tensor X.
  shared_ptr<AbstractTensor> Y;  // Output tensor Y.
};