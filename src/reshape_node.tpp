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
      new_shape[i] = -1;  // Placeholder
    } else if (dim == 0 && allowzero == 1) {
      // If allowzero is set, copy the original input shape at this index
      new_shape[i] = data->get_shape()[i];
      computed_elements *= new_shape[i];
    } else {
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
    computed_elements *= new_shape[inferred_dim_index];  // Update total
  }

  // Apply reshape
  reshaped->reshape(new_shape);

  // Copy the data from the input tensor to the reshaped output tensor
  for (int i = 0; i < data_size; ++i) {
    (*reshaped)[i] = (*data)[i];
  }
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