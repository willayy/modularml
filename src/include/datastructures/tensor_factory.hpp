#pragma once

#include "a_tensor.hpp"
#include "globals.hpp"
#include "mml_array.hpp"
#include "mml_tensor.hpp"
#include "tensor_factory_functions.hpp"

/**
 * @brief A static utility factory class for creating tensors with different
 * shapes and data. Using the setters you can set the function pointers to the
 * tensor constructors. Allowing you to use different tensor implementations.
 */
class TensorFactory {
public:
  // Delete copy constructor and assignment operator.
  TensorFactory(const TensorFactory &) = delete;
  TensorFactory &operator=(const TensorFactory &) = delete;

  /**
   * @brief Creates a tensor with the specified shape and data.
   * @param shape The shape of the tensor to create.
   * @param data The data to fill the tensor with.
   * @return A tensor with the specified shape and data. */
  template <typename T>
  static shared_ptr<Tensor<T>> create_tensor(const array_mml<uli> &shape,
                                             const array_mml<T> &data);

  /**
   * @brief Set the pointer to the tensor constructor function for
   * array_mml<uli> and array_mml<T>.
   * @param tensor_constructor The function pointer to the tensor constructor.
   */
  template <typename T>
  static void set_tensor_constructor_1(
      const std::function<shared_ptr<Tensor<T>>(const array_mml<uli> &shape,
                                                const array_mml<T> &data)>
          &tensor_constructor);

  /**
   * @brief Creates a tensor with the specified shape.
   * @param shape The shape of the tensor to create.
   * @return A tensor with the specified shape. */
  template <typename T>
  static shared_ptr<Tensor<T>> create_tensor(const array_mml<uli> &shape);

  /**
   * @brief Set the pointer to the tensor constructor function for
   * array_mml<uli>.
   * @param tensor_constructor The function pointer to the tensor constructor.
   */
  template <typename T>
  static void set_tensor_constructor_2(
      const std::function<shared_ptr<Tensor<T>>(const array_mml<uli> &shape)>
          &tensor_constructor);

  /**
   * @brief Creates a tensor with the specified shape and data.
   * @param shape The shape of the tensor to create.
   * @param data The data to fill the tensor with.
   * @return A tensor with the specified shape and data. */
  template <typename T>
  static shared_ptr<Tensor<T>> create_tensor(const initializer_list<uli> shape,
                                             const initializer_list<T> data);

  /**
   * @brief Set the pointer to the tensor constructor function for
   * initializer_list<uli> and initializer_list<T>.
   * @param tensor_constructor The function pointer to the tensor constructor.
   */
  template <typename T>
  static void set_tensor_constructor_3(
      const std::function<shared_ptr<Tensor<T>>(
          const initializer_list<uli> shape, const initializer_list<T> data)>
          &tensor_constructor);

  /**
   * @brief Creates a tensor with the specified shape.
   * @param shape The shape of the tensor to create.
   * @return A tensor with the specified shape. */
  template <typename T>
  static shared_ptr<Tensor<T>> create_tensor(const initializer_list<uli> shape);

  /**
   * @brief Set the pointer to the tensor constructor function for
   * initializer_list<uli>.
   * @param tensor_constructor The function pointer to the tensor constructor.
   */
  template <typename T>
  static void set_tensor_constructor_4(
      const std::function<shared_ptr<Tensor<T>>(
          const initializer_list<uli> shape)> &tensor_constructor);

  /**
   * @brief Creates a tensor with the specified shape and data.
   * @param shape The shape of the tensor to create.
   * @param lo_v The lower bound of the random values.
   * @param hi_v The upper bound of the random values.
   * @return A tensor with the specified shape and data. */
  template <typename T>
  static shared_ptr<Tensor<T>> random_tensor(const array_mml<uli> &shape,
                                             T lo_v = T(0), T hi_v = T(1));

private:
  // Private constructor to prevent instantiation.
  TensorFactory() = default;

  // Pointers to the tensor constructor functions.
  template <typename T>
  static std::function<shared_ptr<Tensor<T>>(const array_mml<uli> &shape,
                                             const array_mml<T> &data)>
      tensor_constructor_1;

  template <typename T>
  static std::function<shared_ptr<Tensor<T>>(const array_mml<uli> &shape)>
      tensor_constructor_2;

  template <typename T>
  static std::function<shared_ptr<Tensor<T>>(const initializer_list<uli> shape,
                                             const initializer_list<T> data)>
      tensor_constructor_3;

  template <typename T>
  static std::function<shared_ptr<Tensor<T>>(const initializer_list<uli> shape)>
      tensor_constructor_4;
};

#include "../datastructures/tensor_factory.tpp"