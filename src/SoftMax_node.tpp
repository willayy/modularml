#pragma once

#include "SoftMax_node.hpp"
#include "tensor_utility.hpp"
#include "mml_arithmetic.hpp"
#include <cmath>

/**
 * @brief SoftmaxNode constructor.
 */
template <typename T>
SoftmaxNode<T>::SoftmaxNode(std::shared_ptr<const AbstractTensor> X, int axis)
    : input(X), axis(axis) {}

/**
 * @brief Computes the Softmax function along the specified axis.
 */
template <typename T>
void SoftmaxNode<T>::forward() {
    if (!input) {
        throw std::runtime_error("SoftmaxNode: input tensor is null.");
    }
    // Check if axis is within bounds and transform negative axis as ONNX wants.
    int dim = (axis < 0) ? (input->get_shape().size() + axis) : axis;
    if (dim < 0 || dim >= input->get_shape().size()) {
        throw std::out_of_range("SoftmaxNode: axis is out of range.");
    }

    auto input_mml = std::static_pointer_cast<const Tensor_mml<T>>(input);
    if (!input_mml) {
        throw std::runtime_error("SoftmaxNode: Failed to cast input to Tensor_mml.");
    }

    auto shape = input_mml->get_shape();
    size_t input_size = input_mml->get_size();
    auto result = std::make_shared<Tensor_mml<T>>(shape);

    // Compute max values along the axis
    auto am = std::make_shared<Arithmetic_mml<T>>();
    auto max_vals = am->reduce_max(input_mml, dim);
    auto max_vals_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(max_vals);
    if (!max_vals_mml) {
        throw std::runtime_error("SoftmaxNode: reduce_max failed.");
    }

    // Compute exp(x - max)
    auto temp = std::make_shared<Tensor_mml<T>>(shape);
    int axis_dim = shape[dim];

    size_t stride = 1;
    for (int i = dim + 1; i < shape.size(); ++i) stride *= shape[i];
    size_t group = input_size / axis_dim;

    for (size_t i = 0; i < group; ++i) {
        size_t base = (i / stride) * stride * axis_dim + (i % stride);
        for (int j = 0; j < axis_dim; ++j) {
            T in_val = (*input_mml)[base + j * stride];
            T max_val = (*max_vals_mml)[i];
            (*temp)[base + j * stride] = std::exp(in_val - max_val);
        }
    }

    // Compute sum of exp(x - max)
    auto sum_vals = am->reduce_sum(temp, dim);
    auto sum_vals_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(sum_vals);
    if (!sum_vals_mml) {
        throw std::runtime_error("SoftmaxNode: reduce_sum failed.");
    }

    // Normalize to get softmax values
    for (size_t i = 0; i < group; ++i) {
        size_t base = (i / stride) * stride * axis_dim + (i % stride);
        for (int j = 0; j < axis_dim; ++j) {
            T exp_val = (*temp)[base + j * stride];
            T sum_val = (*sum_vals_mml)[i];
            (*result)[base + j * stride] = exp_val / sum_val;
        }
    }

    output = result;
}

/**
 * @brief Returns the computed output tensor.
 */
template <typename T>
std::shared_ptr<typename SoftmaxNode<T>::AbstractTensor> SoftmaxNode<T>::getOutput() {
    return output;
}

/**
 * @brief Checks if all inputs are properly set.
 */
template <typename T>
bool SoftmaxNode<T>::areInputsFilled() const {
    return input != nullptr;
}

/**
 * @brief Sets the input tensors.
 */
template <typename T>
void SoftmaxNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
    if (inputs.size() == 0) {
        throw std::runtime_error("SoftmaxNode expects at least one input.");
    }
    input = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
}

/**
 * @brief Checks if the output tensor is properly set.
 */
template <typename T>
bool SoftmaxNode<T>::areOutputsFilled() const {
    return output != nullptr;
}

/**
 * @brief Returns an array containing the output tensor.
 */
template <typename T>
array_mml<GeneralDataTypes> SoftmaxNode<T>::getOutputs() const {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(output))};
}