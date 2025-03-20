#pragma once

#include "reshape_node.hpp"

template <typename T>
reshapeNode<T>::reshapeNode(shared_ptr<const AbstractTensor> data, shared_ptr<const Tensor<int64_t>> shape,
                            shared_ptr<AbstractTensor> reshaped, int allowzero)
    : data(data), shape(shape), reshaped(reshaped), allowzero(allowzero) {
  if (allowzero != 0 && allowzero != 1)
    throw runtime_error("Invalid value for allowzero. Must be 0 or 1.");
  if (!shape)
    throw runtime_error("Shape tensor must be of type int64_t.");
  if constexpr (std::is_same_v<T, string> || std::is_same_v<T, bool>) {
    throw runtime_error("Reshape operation is not yet implemented for string or bool tensors.");
  }
}
template <typename T>
void reshapeNode<T>::forward() {
  if (!areInputsFilled())
    throw runtime_error("Reshape inputs are not fully set.");

  if (!data)
    throw runtime_error("Failed to cast data to Tensor<T>.");

  if (!shape)
    throw runtime_error("Failed to cast shape to Tensor<int64_t>.");

  if (!reshaped)
    throw runtime_error("Output tensor reshaped is not allocated.");

  // Determine the size of the shape tensor (number of dimensions for the new shape)
  int shape_size = shape->get_size();

  // Determine the total number of elements in the input data tensor
  int data_size = data->get_size();

  // Create an array to store the new shape values (initialized with same size as shape tensor)
  array_mml<int> new_shape(shape_size);

  // Variables for handling inferred dimension (-1) and computing the total number of elements
  int inferred_dim_index = -1;  // Stores the index of -1 if present
  int computed_elements = 1;    // Tracks the product of explicitly defined shape dimensions

  // Iterate through the shape tensor to determine the new shape values
  for (int i = 0; i < shape_size; ++i) {
    int dim = (*shape)[i];  // Extract the value for the current dimension

    // If dim == -1, mark it for inference (meaning this dimension should be computed)
    if (dim == -1) {
      // Ensure that only one dimension is set to -1 (otherwise, the reshape would be ambiguous)
      if (inferred_dim_index != -1) {
        throw runtime_error("Invalid reshape: multiple -1 values in shape tensor.");
      }
      inferred_dim_index = i;  // Store the index of the inferred dimension
      new_shape[i] = -1;       // Placeholder for now (to be computed later)

      // If dim == 0 and allowzero flag is set, keep the original input shape at this index
    } else if (dim == 0 && allowzero == 1) {
      new_shape[i] = data->get_shape()[i];  // Copy corresponding dimension from input tensor
      computed_elements *= new_shape[i];    // Update total number of elements
    } else {
      // Otherwise, just assign the given dimension value
      new_shape[i] = dim;
      computed_elements *= dim;
    }
  }

  // If -1 was found, infer its value to ensure the total number of elements remains correct
  if (inferred_dim_index != -1) {
    // Ensure that the total number of elements in the new shape matches the original data size
    if (data_size % computed_elements != 0) {
      throw runtime_error("Invalid reshape: inferred dimension does not match total elements.");
    }
    // Compute the missing dimension size and update the new shape
    new_shape[inferred_dim_index] = data_size / computed_elements;

    // Update total computed elements (not strictly necessary, but keeps logic consistent)
    computed_elements *= new_shape[inferred_dim_index];
  }

  // Copy the data from the input tensor to the reshaped output tensor
  *reshaped = *data;

  // Apply reshape with the calculated new shape
  reshaped->reshape(new_shape);
}

template <typename T>
bool reshapeNode<T>::areInputsFilled() const {
  return data && data->get_size() > 0 && shape && shape->get_size() > 0;
}

template <typename T>
void reshapeNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() < 2)
    throw runtime_error("reshapeNode expects two inputs: data and shape.");

  auto valueData = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);
  auto valueShape = std::get_if<shared_ptr<Tensor<int64_t>>>(&inputs[1]);

  if (!valueData || !valueShape)
    throw runtime_error("Failed to cast inputs to the expected tensor types. Check the tensor types.");

  data = *valueData;
  shape = *valueShape;
}

template <typename T>
bool reshapeNode<T>::areOutputsFilled() const {
  if (!reshaped) return false;
  return reshaped->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> reshapeNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(reshaped))};
}