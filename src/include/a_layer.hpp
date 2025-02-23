#pragma once

#include "a_tensor.hpp"      // Direct dependency
#include "a_tensor_func.hpp" // Needed for activation functions

/**
 * @class Layer
 * @brief Abstract base class for neural network layers.
 *
 * This class provides an interface for defining layers in the network.
 * Each layer must define how it manages its tensor representation and
 * its activation function.
 */
template <typename T>
class Layer {
public:
    /// @brief Default constructor.
    explicit Layer() = default;

    /// @brief Virtual destructor ensures proper cleanup in derived classes.
    virtual ~Layer() = default;

    /// @brief Returns the tensor representation of the layer.
    /// @return A Tensor<T> object representing the layer's parameters.
    virtual Tensor<T> tensor() const = 0;

    /// @brief Returns the activation function of the layer.
    /// @return A TensorFunction<T> representing the layer's activation.
    virtual TensorFunction<T> activation() const = 0;

    /// @brief Computes the forward pass through the layer.
    /// @param input The input tensor.
    /// @return The output tensor after applying the layer's transformation.
    virtual Tensor<T> forward(const Tensor<T>& input) const = 0;
};