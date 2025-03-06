#include "include/TanH_node.hpp"

#include <cmath>

#include "include/mml_arithmetic.hpp"
#include "include/mml_tensor.hpp"

void TanHNode::forward() {
  if (!areInputsFilled())
    throw runtime_error("TanHNode inputs are not fully set.");

  visit([this](auto &x_tensor) {
    using T = typename remove_reference_t<decltype(x_tensor)>::value_type;
    auto X_ptr = make_shared<Tensor_mml<T>>(x_tensor);
    Arithmetic_mml<T> arithmetic;
    arithmetic.elementwise_in_place(X_ptr, [](T x) { return std::tanh(x); });
    // Update the output tensor.
    Y->emplace<Tensor_mml<T>>(*X_ptr);
  },
        *X);
}