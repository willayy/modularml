#include "include/ReLU_node.hpp"
#include "include/mml_tensor.hpp"
#include "include/mml_arithmetic.hpp"

void ReLUNode::forward() {
    if (!areInputsFilled())
        throw runtime_error("ReLUNode inputs are not fully set.");

    visit([this](auto &x_tensor) {
        using T = typename remove_reference_t<decltype(x_tensor)>::value_type;

        auto shapeX = x_tensor.get_shape();
        if (shapeX.size() < 2)
            throw runtime_error("Tensor X must be at least 2D.");
        auto X_ptr = make_shared<Tensor_mml<T>>(x_tensor);

        Arithmetic_mml<T> arithmetic;

        arithmetic.elementwise_in_place(X_ptr, [](T x) { return x > 0 ? x : 0; });

        // Update the output tensor.
        Y->template emplace<Tensor_mml<T>>(*X_ptr);
    }, *X);
}