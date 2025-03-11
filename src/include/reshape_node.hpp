#pragma once

#include "a_node.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

/**
 * @class reshapeNode
 * @brief A class representing a reshape node in a computational graph.
 *
 * This class inherits from the Node class and represents the reshape node
 * in a computational graph. It performs resahape after being given an
 * input tensor and a shape tensor.
 */
template <typename T>
class reshapeNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
          std::is_same_v<T, double> ||
          std::is_same_v<T, int32_t> ||
          std::is_same_v<T, int64_t> ||
          std::is_same_v<T, bool> ||
          std::is_same_v<T, string>,
      "reshapeNode supports only float, double, int32_t, int64_t, bool, string");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for ReLUNode.
   *
   * @param data Shared pointer to the tensor data.
   * @param shape Shared pointer to the shape tensor (int64_t).
   * @param reshaped Shared pointer to the reshaped tensor (output).
   * @param allowzero =0 by default. allowzero=1 indicates that if any value in the ‘shape’ input is set to zero, the zero value is honored
   */
  reshapeNode(shared_ptr<const AbstractTensor> data, shared_ptr<const Tensor<int64_t>> shape,
              shared_ptr<AbstractTensor> reshaped, int allowzero = 0)
      : data(data), shape(shape), reshaped(reshaped), allowzero(allowzero) {
    if (allowzero != 0 && allowzero != 1)
      throw runtime_error("Invalid value for allowzero. Must be 0 or 1.");
    if (!shape)
      throw runtime_error("Shape tensor must be of type int64_t.");
  }

  /**
   * @brief Perform the reshape function.
   */
  void forward() override {
    if (!areInputsFilled())
      throw std::runtime_error("Reshape inputs are not fully set.");

    if (!data)
      throw std::runtime_error("Failed to cast data to Tensor_mml<T>.");

    if (!shape)
      throw std::runtime_error("Failed to cast shape to Tensor_mml<int64_t>.");

    if (!reshaped)
      throw std::runtime_error("Output tensor reshaped is not allocated.");

    Arithmetic_mml<T> arithmetic;
    arithmetic.elementwise_in_place(X, [](T x) { return x > 0 ? x : 0; });
    *Y = *X;
  }

  /**
   * @brief Check if the input(s) are filled.
   */
bool areInputsFilled() const override {
    return data && data->get_size() > 0 && shape && shape->get_size() > 0;
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
  // tensors
  shared_ptr<const AbstractTensor> data;  // Input tensor data.
  shared_ptr<const int64_t> shape;       // Input tensor shape.
  shared_ptr<AbstractTensor> reshaped;    // Output tensor reshaped.
  // attributes
  int allowzero;
};