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
   * @brief Constructor for DroputNode.
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
              optional<int> seed = std::nullopt);

  /**
   * @brief Perform the forward pass using dropout.
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   */
  bool areInputsFilled() const override;

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set, where data is inputs[0].
   */
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override;

  /**
   * @brief Check if the output(s) are filled.
   */
  bool areOutputsFilled() const override;

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

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

#include "../Dropout_node.tpp"