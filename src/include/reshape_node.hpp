#pragma once

#include "a_node.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"

/**
 * @class reshapeNode
 * @brief A class representing a reshape node in a computational graph.
 *
 * This class inherits from the Node class and represents the reshape node
 * in a computational graph. It performs reshape after being given an
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
   * @brief Constructor for reshapeNode.
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
    if constexpr (std::is_same_v<T, string> || std::is_same_v<T, bool>) {
      throw runtime_error("Reshape operation is not yet implemented for string or bool tensors.");
    }
  }

  /**
   * @brief Perform the reshape function.
   */
  void forward() override {
    if (!areInputsFilled())
        throw runtime_error("Reshape inputs are not fully set.");

    if (!data)
        throw runtime_error("Failed to cast data to Tensor<T>.");

    if (!shape)
        throw runtime_error("Failed to cast shape to Tensor<int64_t>.");

    if (!reshaped)
        throw runtime_error("Output tensor reshaped is not allocated.");

    int shape_size = shape->get_size();
    int data_size = data->get_size();
    array_mml<int> new_shape(shape_size);

    // Extract shape values from shape tensor
    int inferred_dim_index = -1;
    int computed_elements = 1;

    for (int i = 0; i < shape_size; ++i) {
        int dim = (*shape)[i];

        // If dim == -1, mark it for inference
        if (dim == -1) {
            if (inferred_dim_index != -1) {
                throw runtime_error("Invalid reshape: multiple -1 values in shape tensor.");
            }
            inferred_dim_index = i;
            new_shape[i] = -1; // Placeholder
        }
        else if (dim == 0 && allowzero == 1) {
            // If allowzero is set, copy the original input shape at this index
            new_shape[i] = data->get_shape()[i];
            computed_elements *= new_shape[i];
        }
        else {
            new_shape[i] = dim;
            computed_elements *= dim;
        }
    }

    // Infer missing dimension if -1 is present
    if (inferred_dim_index != -1) {
        if (data_size % computed_elements != 0) {
            throw runtime_error("Invalid reshape: inferred dimension does not match total elements.");
        }
        new_shape[inferred_dim_index] = data_size / computed_elements;
        computed_elements *= new_shape[inferred_dim_index]; // Update total
    }

    // Final validation of reshape
    if (computed_elements != data_size) {
        throw runtime_error("The total number of elements in the new shape does not match the number of elements in the data tensor.");
    }

    // Apply reshape
    reshaped->reshape(new_shape);

    // Copy the data from the input tensor to the reshaped output tensor
    for (int i = 0; i < data_size; ++i) {
        (*reshaped)[i] = (*data)[i];
    }
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
   * @param inputs The input data to be set, where inputs[0] is the data tensor and inputs[1] is the shape tensor.
   */
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override {
    if (inputs.size() < 2)
      throw runtime_error("reshapeNode expects two inputs: data and shape.");

    auto valueData = std::get<shared_ptr<AbstractTensor>>(inputs[0]);
    auto valueShape = std::get<shared_ptr<Tensor<int64_t>>>(inputs[1]);

    if (!valueData || !valueShape)
      throw runtime_error("Failed to cast inputs to the expected tensor types.");

    data = valueData;
    shape = valueShape;
  }

  /**
   * @brief Check if the output(s) are filled.
   */
  bool areOutputsFilled() const override {
    if (!reshaped) return false;
    return reshaped->get_size() > 0;
  }

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(reshaped))};
  }

 private:
  // tensors
  shared_ptr<const AbstractTensor> data;    // Input tensor data.
  shared_ptr<const Tensor<int64_t>> shape;  // Input tensor shape.
  shared_ptr<AbstractTensor> reshaped;      // Output tensor reshaped.

  // attributes
  int allowzero;
};