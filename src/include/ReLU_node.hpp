#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

/**
 * @class ReLUNode
 * @brief A class representing a ReLUNode node in a computational graph.
 *
 * This class inherits from the Node class and represents the rectified linear function (ReLU) node
 * in a computational graph. It performs the forward pass computation appling ReLU elementwise.
 */
class ReLUNode : public Node {
 public:
  // Type constraints: no bfloat16 or float16 for now (not native to c++ 17).
  using DataTypes = variant<
      Tensor_mml<float>,
      Tensor_mml<double>,
      Tensor_mml<int32_t>,
      Tensor_mml<int64_t>,
      Tensor_mml<uint32_t>,
      Tensor_mml<uint64_t>>;

  /**
   * @brief Constructor for ReLUNode.
   *
   * @param X Shared pointer to the tensor A.
   * @param Y Shared pointer to the output tensor.
   */
  ReLUNode(shared_ptr<DataTypes> X, shared_ptr<DataTypes> Y)
      : X(X), Y(Y) {}

  /**
   * @brief Perform the forward pass computation using ReLU activation function.
   *
   * This function performs the forward pass computation using ReLU.
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   *
   * @return True if the input(s) are filled, false otherwise.
   */
  bool areInputsFilled() const override {
    return X && visit([](const auto& t) { return t.get_size() > 0; }, *X);
  }

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set, where A is inputs[0].
   */
  void setInputs(const vector<GeneralDataTypes>& inputs) override {
    if (inputs.size() < 1)
      throw runtime_error("ReLUNode expects at least one input: X.");

    // Deduce type from the first input.
    visit([this, &inputs](const auto& tensorA) {
      using T = typename remove_reference_t<decltype(tensorA)>::value_type;

      // Restrict T to allowed types.
      if constexpr (!(std::is_same_v<T, float> ||
                      std::is_same_v<T, double> ||
                      std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, uint64_t>)) {
        throw runtime_error("ReLUNode input type not supported.");
      } else {
        try {
          auto valueX = std::get<Tensor_mml<T>>(inputs[0]);

          X->template emplace<Tensor_mml<T>>(valueX);

        } catch (const std::bad_variant_access&) {
          throw runtime_error("Data type mismatch: All inputs must have the same type as X.");
        }
      }
    },
          inputs[0]);
  }

  /**
   * @brief Check if the output(s) are filled.
   *
   * @return True if the output(s) are filled, false otherwise.
   */
  bool areOutputsFilled() const override {
    return Y && visit([](const auto& t) { return t.get_size() > 0; }, *Y);
  }

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  vector<GeneralDataTypes> getOutputs() const override {
    if (!Y) {
      throw runtime_error("Output tensor Y is not filled!");
    }
    return {visit([](const auto& arg) -> GeneralDataTypes { return arg; }, *Y)};
  }

 private:
  // Inputs
  shared_ptr<DataTypes> X;  // Input tensor X.

  // Output
  shared_ptr<DataTypes> Y;  // Output tensor Y.
};