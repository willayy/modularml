#pragma once

#include "SoftMax_node.hpp"
#include "tensor_utility.hpp"

/**
 * @brief SoftmaxNode constructor.
 */
template <typename T>
SoftmaxNode<T>::SoftmaxNode(std::shared_ptr<const AbstractTensor> X, std::shared_ptr<AbstractTensor> Y, int axis)
    : input_(X), output_(Y), axis_(axis) {}

/**
 * @brief Computes the Softmax function along the specified axis.
 */
template <typename T>
void SoftmaxNode<T>::forward() {
    auto& in = *input_;
    auto& out = *output_;

    // Ensure input_ and output_ are allocated
    if (!input_ || !output_) {
        throw std::runtime_error("SoftmaxNode: input or output tensor is null.");
    }

    int dim = (axis_ < 0) ? (in.get_shape().size() + axis_) : axis_;

    // Compute max values along the specified axis
    auto max_vals = tensor_utility::reduce_max(in, dim);

    // Create arithmetic module
    const std::shared_ptr<ArithmeticModule<T>> am = std::make_shared<Arithmetic_mml<T>>();

    // ✅ FIX: Convert shape to std::initializer_list<int> so tensor_mml_p<T>() accepts it
    auto temp = tensor_mml_p<T>({in.get_shape().begin(), in.get_shape().end()});

    // Compute exp(input - max_vals)
    am->elementwise_in_place(temp, [](T x) { return std::exp(x); });

    // Compute sum of exponentials along the axis
    auto sum_vals = tensor_utility::reduce_sum(temp, dim);

    // Element-wise division to normalize
    am->elementwise(temp, [](T x) { return x; }, sum_vals);

    // Assign the result to the output
    output_ = temp;
}

/**
 * @brief Returns the output tensor.
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
    if (inputs.size() < 1) {
        throw std::runtime_error("SoftmaxNode expects at least one input.");
    }
    
    input_ = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
}

/**
 * @brief Checks if all outputs are properly set.
 */
template <typename T>
bool SoftmaxNode<T>::areOutputsFilled() const {
    return output_ != nullptr;
}

/**
 * @brief Returns an array of the output tensors.
 */
template <typename T>
array_mml<GeneralDataTypes> SoftmaxNode<T>::getOutputs() const {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(output_))};
}