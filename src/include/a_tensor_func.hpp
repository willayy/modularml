#ifndef A_TENSOR_FUNC_HPP
#define A_TENSOR_FUNC_HPP

#include "tensor.hpp"

/**
 * @class TensorFunction
 * @brief Abstract TensorFunction class/type.
 *
 * This class provides an interface for tensor functions, including methods
 * for computing the function, its derivative, and its primitive.
 */

/// @brief Abstract TensorFunction class/type.
/// @tparam T the type of the data contained in the tensor. E.g. int, float,
template <typename T>
class TensorFunction {
public:
    /// @brief Default constructor for TensorFunction.
    explicit TensorFunction() = default;

    /// @brief Virtual destructor for TensorFunction.
    /// @details Ensures derived class destructors are called properly.
    virtual ~TensorFunction() = default;

    // To be implemented by derived classes:

    /// @brief Apply func to the tensor.
    /// @param t the tensor to apply the function to.    
    virtual Tensor<T> func(const Tensor<T>& t) const = 0;

    /// @brief Apply the derivative of the function to the tensor.
    /// @param t the tensor to apply the function to.
    virtual Tensor<T> derivative(const Tensor<T>& t) const = 0;

    /// @brief Apply the primitive of the function to the tensor.
    /// @details The indefinite integral of the function applied element-wise to the tensor
    /// @param t the tensor to apply the function to.
    virtual Tensor<T> primitive(const Tensor<T>& t) const = 0;
};

#endif