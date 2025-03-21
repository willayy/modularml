#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

template <typename T> class ELUNode : public Node {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "ELUNode supports only float, double");

public:
  using AbstractTensor = Tensor<T>;
  ELUNode(shared_ptr<AbstractTensor> X, shared_ptr<AbstractTensor> Y,
          float alpha = 1.0f);

  /**
   * @brief Perform the forward pass computation using the ELU function.
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   * @return True if the input(s) are filled, false otherwise.
   */
  bool areInputsFilled() const override;

  /**
   * @brief Set the input(s) for the node.
   * @param inputs The input data to be set.
   */
  void setInputs(const array_mml<GeneralDataTypes> &inputs)
      override; // This function could have better
                // type safety somehow maybe.

  /**
   * @brief Check if the output(s) are filled.
   * @return True if the output(s) are filled, false otherwise.
   */
  bool areOutputsFilled() const override;

  /**
   * @brief Get the output of the node.
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

private:
  shared_ptr<AbstractTensor> X;
  shared_ptr<AbstractTensor> Y;
  float alpha;
}