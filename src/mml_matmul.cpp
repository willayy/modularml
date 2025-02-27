#include "mml_matmul.hpp"

template <typename T>
MatMul<T>::MatMul(const Tensor<T>& weight) : weight(weight) {}

template <typename T>
std::unique_ptr<Tensor<T>> MatMul<T>::forward(const Tensor<T>& input) const {
    auto result = input * this->weight;
    std::unique_ptr<Tensor<T>> result_ptr = std::make_unique<Tensor<T>>(result);
    return result_ptr;
}

/**
 * @brief This function is not needed, thus left empty
 */
template <typename T>
Tensor<T> MatMul<T>::tensor() const {
    return Tensor<T>();
}

/**
 * @brief This function is not needed, thus left empty
 */
template <typename T>
std::unique_ptr<TensorFunction<T>> MatMul<T>::activation() const {
    return nullptr;
}


