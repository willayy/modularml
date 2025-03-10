#pragma once

#include <cmath>
#include <stdexcept>
#include <variant>
#include <limits>
#include <iostream>

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"
#include "mml_tensor.hpp"

/**
 * @class SoftmaxNode
 * @brief A class representing a Softmax node in a computational graph.
 *
 * This class inherits from the Node class and represents the Softmax
 * activation in a computational graph. It performs the forward
 * pass computation applying Softmax along a specified dimension.
 */
template <typename T>
class SoftmaxNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
          std::is_same_v<T, double>,
      "SoftmaxNode supports only float, double");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for SoftmaxNode.
   *
   * @param X Shared pointer to the input tensor X.
   * @param Y Shared pointer to the output tensor Y.
   * @param dim The dimension along which to apply Softmax.
   */
  SoftmaxNode(std::shared_ptr<AbstractTensor> X,
              std::shared_ptr<AbstractTensor> Y,
              int dim = -1)  // Default to last dimension
      : X(X), Y(Y), dim_(dim) {}

  /**
   * @brief Perform the forward pass computation applying Softmax.
   */
  void forward() override {
    if (!areInputsFilled())
        throw std::runtime_error("SoftmaxNode inputs are not fully set.");

    if (!X || !Y)
        throw std::runtime_error("Input or Output tensor is not allocated.");

    auto shape = X->get_shape();
    
    std::cout << "SoftmaxNode Input Shape: ";
    for (auto s : shape) std::cout << s << " ";
    std::cout << std::endl;

    // Ensure valid dimension
    if (dim_ < 0) dim_ += shape.size();
    if (dim_ < 0 || dim_ >= static_cast<int>(shape.size()))
        throw std::runtime_error("Invalid dimension for Softmax.");

    std::cout << "Softmax applied along dimension: " << dim_ << std::endl;

    int dim_size = shape[dim_];
    if (dim_size <= 0) {
        throw std::runtime_error("ERROR: Invalid dim_size for Softmax: " + std::to_string(dim_size));
    }
    int outer_size = 1, inner_size = 1;

    // Compute sizes for iteration
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            throw std::runtime_error("ERROR: Invalid shape dimension at index " + std::to_string(i) +
                                     ": " + std::to_string(shape[i]));
        }
    }
    for (int i = 0; i < dim_; i++) outer_size *= shape[i];
    for (int i = dim_ + 1; i < shape.size(); i++) inner_size *= shape[i];

    std::cout << "outer_size: " << outer_size << ", inner_size: " << inner_size << ", dim_size: " << dim_size << std::endl;

    auto max_vals = std::make_shared<Tensor_mml<T>>(shape);
    auto exp_vals = std::make_shared<Tensor_mml<T>>(shape);
    auto sum_exp = std::make_shared<Tensor_mml<T>>(shape);

    // Compute max per slice along the chosen dimension
    for (int outer = 0; outer < outer_size; ++outer) {
        for (int inner = 0; inner < inner_size; ++inner) {
            T max_val = std::numeric_limits<T>::lowest();
            for (int j = 0; j < dim_size; ++j) {
                int idx = outer * dim_size * inner_size + j * inner_size + inner;
                if (idx < 0 || idx >= static_cast<int>(X->get_size())) {
                    throw std::runtime_error("Index out of bounds in Softmax computation: " + std::to_string(idx));
                }
                max_val = std::max(max_val, (*X)[idx]);
            }

            for (int j = 0; j < dim_size; ++j) {
                int idx = outer * dim_size * inner_size + j * inner_size + inner;
                if (idx < 0 || idx >= static_cast<int>(X->get_size())) {
                    throw std::runtime_error("Index out of bounds in Softmax computation: " + std::to_string(idx));
                }
                (*max_vals)[idx] = max_val;
            }
        }
    }

    // Compute exponentials using max subtraction (log-sum-exp trick)
    for (int i = 0; i < X->get_size(); ++i) {
        if (i < 0 || i >= X->get_size()) continue;
        T shifted_val = (*X)[i] - (*max_vals)[i];
        (*exp_vals)[i] = std::exp(std::max(shifted_val, static_cast<T>(-40))); // Prevent underflow
    }

    // Compute sum of exponentials per slice
    for (int outer = 0; outer < outer_size; ++outer) {
        for (int inner = 0; inner < inner_size; ++inner) {
            T sum_val = 0;
            for (int j = 0; j < dim_size; ++j) {
                int idx = outer * dim_size * inner_size + j * inner_size + inner;
                if (idx < 0 || idx >= static_cast<int>(X->get_size())) {
                    throw std::runtime_error("Index out of bounds in Softmax computation: " + std::to_string(idx));
                }
                sum_val += (*exp_vals)[idx];
            }

            for (int j = 0; j < dim_size; ++j) {
                int idx = outer * dim_size * inner_size + j * inner_size + inner;
                if (idx < 0 || idx >= static_cast<int>(X->get_size())) {
                    throw std::runtime_error("Index out of bounds in Softmax computation: " + std::to_string(idx));
                }
                (*sum_exp)[idx] = std::max(sum_val, static_cast<T>(1e-8)); // Avoid division by zero
            }
        }
    }

    // Normalize
    for (int i = 0; i < X->get_size(); ++i) {
        if (i < 0 || i >= X->get_size()) continue;
        (*Y)[i] = (*exp_vals)[i] / ((*sum_exp)[i] + static_cast<T>(1e-8)); // Prevent division by zero
    }

    std::cout << "Final Softmax Output: ";
    for (int i = 0; i < X->get_size(); ++i) {
        std::cout << "Index " << i << ": " << (*Y)[i] << std::endl;
    }
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
      throw std::runtime_error("SoftmaxNode expects at least one input: X.");

    auto valueX = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);

    auto valueX_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueX);
    if (!X || !valueX_mml)
      throw std::runtime_error("Failed to cast X or input X to Tensor_mml<T>.");
    *X = *valueX_mml;
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
  std::shared_ptr<AbstractTensor> X;  // Input tensor X.
  std::shared_ptr<AbstractTensor> Y;  // Output tensor Y.
  int dim_;  // Dimension along which Softmax is applied.
};