#include "mml_matmul.hpp"

template class MatMul<float>;
template class MatMul<double>;

/**
 * @brief Constructs a MatMul layer with the given weight tensor.
 *
 * Initializes the MatMul layer with the provided weight tensor, which will be used
 * during the forward pass for matrix multiplication.
 *
 * @param weight The weight tensor used in the matrix multiplication.
 */
template <typename T>
MatMul<T>::MatMul(const Tensor<T>& weight) : weight(weight) {}

/**
 * @brief Performs the forward pass of the MatMul layer.
 *
 * Multiplies the input tensor by the weight tensor and returns the result as a new tensor.
 *
 * @param input The input tensor to be multiplied.
 * @return A unique pointer to the resulting tensor.
 */
template <typename T>
Tensor<T> MatMul<T>::forward(const Tensor<T>& input) const {
    return input * weight;
}

/**
 * @brief This function is not needed, thus just returns an empty tensor
 */
template <typename T>
Tensor<T> MatMul<T>::tensor() const {
    return weight;
}

/**
 * @brief This function is not needed, thus just returns a nullptr
 */
template <typename T>
std::unique_ptr<TensorFunction<T>> MatMul<T>::activation() const {
    return nullptr;
}
