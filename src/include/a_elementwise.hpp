#pragma once

#include <modularml>

#define ASSERT_ALLOWED_TYPES(T) static_assert(std::is_arithmetic_v<T>, "Data structure type must be an arithmetic type.")

/**
 * @class Elementwise
 * @brief Abstract base class for element-wise operations on tensors.
 *
 * This class provides an interface for applying a function element-wise to a
 * tensor.
 *
 * @tparam T The data type of the elements in the tensor.
 */
template <typename T>
class Elementwise {
 public:
  /**
   * @brief Apply a function element-wise to a tensor.
   *
   * This pure virtual function must be implemented by derived classes to apply
   * a given function to each element of the input tensor.
   *
   * @param t The tensor to which the function will be applied.
   * @param f A pointer to the function to be applied to each element of the
   * tensor.
   * @return A new modified tensor.
   */
  virtual shared_ptr<Tensor<T>> apply(const shared_ptr<Tensor<T>> t, T (*f)(T)) = 0;
};