#pragma once

#include "Dropout_node.hpp"

template <typename T>
DropoutNode<T>::DropoutNode(shared_ptr<const AbstractTensor> data,
                            shared_ptr<AbstractTensor> output,
                            optional<shared_ptr<AbstractTensor>> mask,
                            float ratio,
                            bool training_mode,
                            optional<int> seed)
    : data(data), output(output), mask(mask), ratio(ratio), training_mode(training_mode), seed(seed) {}

template <typename T>
void DropoutNode<T>::forward() {
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

template <typename T>
bool DropoutNode<T>::areInputsFilled() const {
  return data && data->get_size() > 0 &&
         (!mask.has_value() || (mask.value() && mask.value()->get_size() > 0));
}

template <typename T>
void DropoutNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
  if (inputs.size() < 1)
    throw runtime_error("DropoutNode expects at least one input: input.");

  auto valueData = std::get_if<shared_ptr<AbstractTensor>>(&inputs[0]);

  if (!valueData)
    throw runtime_error("Failed to cast Input to the expected tensor types.");
  data = std::const_pointer_cast<AbstractTensor>(*valueData);
}

template <typename T>
bool DropoutNode<T>::areOutputsFilled() const {
  if (!output) return false;
  return output->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> DropoutNode<T>::getOutputs() const {
  return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(output))};
}