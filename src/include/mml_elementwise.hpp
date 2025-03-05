#pragma once

#include <modularml>

#define ASSERT_ALLOWED_TYPES(T) static_assert(std::is_arithmetic_v<T>, "Data structure type must be an arithmetic type.")

/**
 * @brief Concrete implementation of Elementwise for applying functions
 *   element-wise.
 *
 * This class provides an implementation of the abstract method from the
 * Elementwise interface, applying a given function element-wise to a tensor.
 * It can be used with tensors of any type.
 *
 * @tparam T The type of the tensor elements.
 */
template <typename T>
class mml_elementwise : public Elementwise<T> {
 public:
  // This function can be made way more efficent by the use of multi-threading
  // I intend on making that an improvement in the future
  /**
   * @brief Applies a given function element-wise to a tensor.
   *
   * This function iterates over each element of the input tensor and applies the provided function to it.
   *
   * @tparam T The data type of the elements in the tensor.
   * @param t The input tensor to which the function will be applied.
   * @param f A pointer to the function that will be applied to each element of the tensor.
   * @return A modified tensor after applying the function.
   */
  shared_ptr<Tensor<T>> apply(const shared_ptr<Tensor<T>> t, T (*f)(T)) override {
    auto tensor = t;
    for (int i = 0; i < tensor->get_shape()[0]; i++) {
      for (int j = 0; j < tensor->get_shape()[1]; j++) {
        (*tensor)[{i, j}] = f((*tensor)[{i, j}]);
      }
    }
    return tensor;
  }
};
