#pragma once

#include "a_node.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

template <typename T>
class AddNode : Node {
  static_assert(
      std::is_same_v<T, float> ||
          std::is_same_v<T, double> ||
          std::is_same_v<T, int32_t> ||
          std::is_same_v<T, int64_t>,
      "AddNode supports only float, double, int32_t, int64_t");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for AddNode.
   *
   * @param A Shared pointer to the first input tensor.
   * @param B Shared pointer to the second input tensor.
   * @param C Shared pointer to the output tensor.
   */
  AddNode(shared_ptr<const AbstractTensor> A, shared_ptr<const AbstractTensor> B,
          shared_ptr<AbstractTensor> C);

  /**
   * @brief Performs element-wise binary addition in the two input tensors and stores the result in the output tensor.
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   */
  bool areInputsFilled() const override;

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set, where inputs[0] is the data tensor and inputs[1] is the shape tensor.
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
  // tensors
  shared_ptr<const AbstractTensor> A;  // Input tensor A
  shared_ptr<const AbstractTensor> B;  // Input tensor B
  shared_ptr<AbstractTensor> C;        // Output tensor C

  /**
   * @brief Helper function used when broadcasting addition is required.
   * Likely only temporary to be replaced with something that can be used in multiple nodes instead.
   *
   * @return The output data.
   */
  void broadcast_addition() const;
};

#include "../Add_node.tpp"