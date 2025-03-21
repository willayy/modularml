#pragma once

#include "SoftMax_node.hpp"
#include "tensor_utility.hpp"

/**
 * @brief SoftmaxNode constructor.
 */
template <typename T>
SoftmaxNode<T>::SoftmaxNode(std::shared_ptr<const AbstractTensor> X, int axis)
    : input_(X), axis_(axis) {}

/**
 * @brief Computes the Softmax function along the specified axis.
 */
template <typename T>
void SoftmaxNode<T>::forward() {
    if (!input_) {
        throw std::runtime_error("SoftmaxNode: input tensor is null.");
    }

    // Adjust negative axis
    int dim = (axis_ < 0) ? (input_->get_shape().size() + axis_) : axis_;

    // Validate axis
    if (dim < 0 || dim >= input_->get_shape().size()) {
        throw std::out_of_range("SoftmaxNode: axis is out of range.");
    }

    // Cast input_ to Tensor_mml<T>
    auto input_mml = std::static_pointer_cast<const Tensor_mml<T>>(input_);
    if (!input_mml) {
        throw std::runtime_error("SoftmaxNode: Failed to cast input to Tensor_mml.");
    }

    // Compute softmax
    auto am = std::make_shared<Arithmetic_mml<T>>();
    output_ = am->elementwise_softmax(input_mml, dim);
}

/**
 * @brief Returns the computed output tensor.
 */
template <typename T>
std::shared_ptr<typename SoftmaxNode<T>::AbstractTensor> SoftmaxNode<T>::getOutput() {
    return output_;
}

/**
 * @brief Checks if all inputs are properly set.
 */
template <typename T>
bool SoftmaxNode<T>::areInputsFilled() const {
    return input_ != nullptr;
}

/**
 * @brief Sets the input tensors.
 */
template <typename T>
void SoftmaxNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
    if (inputs.size() == 0) {
        throw std::runtime_error("SoftmaxNode expects at least one input.");
    }
    input_ = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
}

/**
 * @brief Checks if the output tensor is properly set.
 */
template <typename T>
bool SoftmaxNode<T>::areOutputsFilled() const {
    return output_ != nullptr;
}

/**
 * @brief Returns an array containing the output tensor.
 */
template <typename T>
array_mml<GeneralDataTypes> SoftmaxNode<T>::getOutputs() const {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(output_))};
}