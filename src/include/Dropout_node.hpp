#pragma once

#include "a_node.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

/**
 * @class DropoutNode
 * @brief A class representing a Dropout node in a computational graph.
 *
 * This class inherits from the Node class and represents the Dropout node
 * in a computational graph. It performs dropout only if training is set to True.
 */
template <typename T>
class DropoutNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
          std::is_same_v<T, double>,
      "DropoutNode supports only float, double");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for TanHNode.
   *
   * @param data Shared pointer to the input tensor data.
   * @param output Shared pointer to the output tensor output.
   * @param mask Optional shared pointer to the output tensor mask.
   * @param ratio Dropout ratio, 0.5 by default.
   * @param training_mode Training mode, False by default.
   * @param seed Random seed, None by default.
   */
  DropoutNode(shared_ptr<const AbstractTensor> data,
              shared_ptr<AbstractTensor> output,
              optional<shared_ptr<AbstractTensor>> mask = std::nullopt,
              float ratio = 0.5,
              bool training_mode = false,
              optional<int> seed = std::nullopt)
      : data(data), output(output), mask(mask), ratio(ratio), training_mode(training_mode), seed(seed) {}

  /**
   * @brief Perform the forward pass using dropout.
   */
  void forward() override {
    if (!areInputsFilled())
      throw runtime_error("DropoutNode inputs are not fully set.");

    if (!data)
      throw runtime_error("Failed to cast data to Tensor_mml<T>.");

    if (!output)
      throw runtime_error("Output tensor output is not allocated.");

    if (data->get_shape().size() < 1)
      throw runtime_error("Tensor data must be at least 1D.");

    if (training_mode) {
      throw runtime_error("DropoutNode forward pass in training mode is not implemented yet.");
    } else {
      *output = *data;
    }
  }

  /**
   * @brief Check if the input(s) are filled.
   */
  bool areInputsFilled() const override {
    return data && data->get_size() > 0 &&
           (!mask.has_value() || (mask.value() && mask.value()->get_size() > 0));
  }

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set, where data is inputs[0].
   */
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override {
    if (inputs.size() < 1)
      throw runtime_error("DropoutNode expects at least one input: input.");

    auto valueData = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);

    if (!valueData)
      throw runtime_error("Failed to cast Input to the expected tensor types.");
    data = std::const_pointer_cast<AbstractTensor>(*valueData);
  }

  /**
   * @brief Check if the output(s) are filled.
   */
  bool areOutputsFilled() const override {
    if (!output) return false;
    return output->get_size() > 0;
  }

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(output))};
  }

 private:
  // Inputs
  shared_ptr<const AbstractTensor> data;  // Input tensor.

  // Outputs
  shared_ptr<AbstractTensor> output;          // Output tensor.
  optional<shared_ptr<AbstractTensor>> mask;  // Optional output tensor mask.

  // Attributes
  float ratio;         // Dropout ratio.
  bool training_mode;  // Training mode.
  optional<int> seed;  // Random seed.
};