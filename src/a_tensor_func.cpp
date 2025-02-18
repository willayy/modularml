#include "a_tensor_func.hpp"

/**
 * @class ExampleTensorFunction
 * @brief A class that implements a tensor function with its derivative and primitive.
 *
 * This class provides implementations of tensor functions. Calculating the derivative, 
 * the primitive and applying func.
 */
class ExampleTensorFunction : public TensorFunction<float> {
public:
    // Implement the function
    virtual Tensor<float> func(const Tensor<float>& t) const override {
        // Implement the function here
        return t;
    }

    // Implement the derivative
    virtual Tensor<float> derivative(const Tensor<float>& t) const override {
        // Implement the derivative here
        return t;
    }

    // Implement the primitive
    virtual Tensor<float> primitive(const Tensor<float>& t) const override {
        // Implement the primitive here
        return t;
    }
};

// Factory function for creating an instance, not 100% sure if we want to keep this
TensorFunction<float>* create_example_tensor_function() {
    return new ExampleTensorFunction();
}
