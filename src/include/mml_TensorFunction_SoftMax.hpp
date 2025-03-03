#pragma once

#include "tensor.hpp"
#include "a_tensor_func.hpp"
#include "mml_tensors.hpp"
#include <cmath>
#include <limits>

/**
 * @class mml_TensorFunction_SoftMax
 * @brief Implements the SoftMax activation function for tensors.
 */
class mml_TensorFunction_SoftMax : public TensorFunction<float> {
private:
    int axis; // Axis to apply SoftMax on

public:
    /**
     * @brief Constructor
     * @param axis The axis along which to apply SoftMax (default -1 = last axis)
     */
    explicit mml_TensorFunction_SoftMax(int axis = -1) : axis(axis) {}

    /**
     * @brief Apply SoftMax to a tensor along the specified axis.
     */
    Tensor<float> func(const Tensor<float>& t) const override {
        auto shape = t.get_shape();
        int adjusted_axis = (axis < 0) ? (shape.size() + axis) : axis;

        if (adjusted_axis < 0 || adjusted_axis >= static_cast<int>(shape.size())) {
            throw std::out_of_range("Invalid axis for SoftMax function");
        }

        Tensor<float> output(t.get_shape()); // Zero-initialized tensor
        int axis_size = shape[adjusted_axis];
        int batch_size = t.get_size() / axis_size; // Number of rows

        // **1D SoftMax**
        if (shape.size() == 1) { 
            float max_val = t.max_element();
            float sum = 0.0f;

            // Compute exponentials and sum
            for (int i = 0; i < t.get_size(); i++) {
                output[i] = std::exp(t[i] - max_val);
                sum += output[i];
            }

            // Normalize
            for (int i = 0; i < t.get_size(); i++) {
                output[i] /= sum;
            }
        } 
        // **2D Row-Wise SoftMax**
        else { 
            for (int i = 0; i < batch_size; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                float sum = 0.0f;

                // Step 1: Find max value in row
                for (int j = 0; j < axis_size; ++j) {
                    max_val = std::max(max_val, t[{i, j}]);
                }

                // Step 2: Compute exponentials & sum
                for (int j = 0; j < axis_size; ++j) {
                    output[{i, j}] = std::exp(t[{i, j}] - max_val);
                    sum += output[{i, j}];
                }

                // Step 3: Normalize row-wise
                for (int j = 0; j < axis_size; ++j) {
                    output[{i, j}] /= sum;
                }
            }
        }

        return output;
    }

    /**
     * @brief Compute the derivative of SoftMax.
     */
    Tensor<float> derivative(const Tensor<float>& t) const override {
        Tensor<float> softmax_t = func(t); // Get SoftMax output
        Tensor<float> jacobian({t.get_size(), t.get_size()});

        for (int i = 0; i < t.get_size(); ++i) {
            for (int j = 0; j < t.get_size(); ++j) {
                if (i == j) {
                    jacobian[{i, j}] = softmax_t[i] * (1 - softmax_t[i]);
                } else {
                    jacobian[{i, j}] = -softmax_t[i] * softmax_t[j];
                }
            }
        }
        return jacobian;
    }

    /**
     * @brief Compute the primitive function (log of SoftMax).
     */
    Tensor<float> primitive(const Tensor<float>& t) const override {
        Tensor<float> result = t;
        result.apply_function([](float x) { return std::log(x); }); // Use apply_function()
        return result;
    }
};