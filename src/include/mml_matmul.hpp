#pragma once

#include "a_layer.hpp"
#include "globals.hpp"
#include "mml_gemm.hpp"
#include "mml_tensors.hpp"

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
    MatMul(const Tensor<T>& weight) : weight(weight){};

    /**
     * @brief Performs the forward pass of the MatMul layer.
     *
     * This method multiplies the input tensor by the weight tensor to produce the result.
     *
     * @param input The input tensor to be multiplied.
     * @return A unique pointer to the result tensor.
     */
    Tensor<T> forward(const Tensor<T>& input) const override {
        auto A = make_shared<Tensor<T>>(input);
        auto B = make_shared<Tensor<T>>(this->weight);
        Tensor<T> result = tensor_mml<T>({input.get_shape().at(0), weight.get_shape().at(1)});
        auto result_ptr = make_shared<Tensor<T>>(result);

        shared_ptr<GemmModule<T>> gemm = make_shared<Gemm_mml<T>>();
        gemm->gemm_inner_product(
            0, 0,
            input.get_shape().at(0), weight.get_shape().at(1), weight.get_shape().at(0),
            1.0f,
            make_shared<Tensor<T>>(input), input.get_shape().at(1),
            make_shared<Tensor<T>>(weight), weight.get_shape().at(1),
            0.0f,
            result_ptr, result_ptr->get_shape().at(1));

        return *result_ptr;
    };

    /**
     * @brief Returns the weight tensor used in the layer.
     *
     * This function returns the weight tensor. It is a placeholder for now.
     *
     * @return A tensor object representing the weights.
     */
    Tensor<T> tensor() const override {
        return weight;
    };

    /**
     * @brief Returns the activation function for the layer.
     *
     * This function is used for layers with activation functions. For MatMul, it is not implemented.
     *
     * @return A nullptr since MatMul does not have an activation function.
     */
    std::unique_ptr<TensorFunction<T>> activation() const override {
        return nullptr;
    };

    /**
     * @brief Returns the MatMul instance's weight tensor
     *
     * @return A Tensor object
     */
    Tensor<T> get_weights() const {
        return this->weight;
    }

   private:
    Tensor<T> weight;  // The weight tensor used in the matrix multiplication
};