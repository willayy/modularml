#pragma once

#include "a_layer.hpp"
#include "globals.hpp"

/**
 * @class MatMul
 * @brief Represents a matrix multiplication layer.
 *
 * This class implements a MatMul layer for neural networks, where the forward pass
 * performs matrix multiplication between the input tensor and a weight tensor.
 *
 * The forward pass computes the result by multiplying the input tensor by the weight tensor.
 *
 * @tparam T The data type of the tensor elements (e.g., float, double).
 */
template <typename T>
class MatMul : public Layer<T> {
   public:
    /**
     * @brief Constructs a MatMul layer with the given weight tensor.
     *
     * @param weight The weight tensor used in the matrix multiplication.
     */
    MatMul(const Tensor<T>& weight);

    /**
     * @brief Performs the forward pass of the MatMul layer.
     *
     * This method multiplies the input tensor by the weight tensor to produce the result.
     *
     * @param input The input tensor to be multiplied.
     * @return A unique pointer to the result tensor.
     */
    Tensor<T> forward(const Tensor<T>& input) const override;

    /**
     * @brief Returns the weight tensor used in the layer.
     *
     * This function returns the weight tensor. It is a placeholder for now.
     *
     * @return A tensor object representing the weights.
     */
    Tensor<T> tensor() const override;

    /**
     * @brief Returns the activation function for the layer.
     *
     * This function is used for layers with activation functions. For MatMul, it is not implemented.
     *
     * @return A nullptr since MatMul does not have an activation function.
     */
    std::unique_ptr<TensorFunction<T>> activation() const override;

   private:
    Tensor<T> weight;  // The weight tensor used in the matrix multiplication
};