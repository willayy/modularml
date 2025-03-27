#pragma once

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"
#include "mml_tensor.hpp"

template <typename T> class TensorFactory {

  /**
   * @brief Get the instance of the TensorFactory.
   * @return The instance of the TensorFactory. */
  static TensorFactory &getInstance();

  // Delete copy constructor and assignment operator.
  TensorFactory(const TensorFactory &) = delete;
  TensorFactory &operator=(const TensorFactory &) = delete;

  /**
   * @brief Creates a tensor with the specified shape and data.
   * @param shape The shape of the tensor to create.
   * @param data The data to fill the tensor with.
   * @return A tensor with the specified shape and data. */
  shared_ptr<Tensor<T>> create_tensor(const array_mml<uli> &shape,
                                      const array_mml<T> &data) const;

  /**
   * @brief Creates a tensor with the specified shape.
   * @param shape The shape of the tensor to create.
   * @return A tensor with the specified shape. */
  shared_ptr<Tensor<T>> create_tensor(const array_mml<uli> &shape) const;

  /**
   * @brief Creates a tensor with the specified shape and data.
   * @param shape The shape of the tensor to create.
   * @param data The data to fill the tensor with.
   * @return A tensor with the specified shape and data. */
  shared_ptr<Tensor<T>> create_tensor(const initializer_list<uli> shape,
                                      const initializer_list<T> data) const;

  /**
   * @brief Creates a tensor with the specified shape.
   * @param shape The shape of the tensor to create.
   * @return A tensor with the specified shape. */
  shared_ptr<Tensor<T>> create_tensor(const initializer_list<uli> shape) const;

  /**
   * @brief Creates a tensor with the specified shape and data.
   * @param shape The shape of the tensor to create.
   * @param lo_v The lower bound of the random values.
   * @param hi_v The upper bound of the random values.
   * @return A tensor with the specified shape and data. */
  shared_ptr<Tensor<T>> random_tensor(const array_mml<uli> &shape,
                                      T lo_v = T(0), T hi_v = T(1)) const;

  /**
   * @brief Creates a random array with integral values.
   * @param shape The shape of the tensor to create.
   * @param lo_sz The lower bound of the array size.
   * @param hi_sz The upper bound of the array size.
   * @param lo_v The lower bound of the random values.
   * @param hi_v The upper bound of the random values.
   * @return A tensor with the specified shape and data. */
  array_mml<T> random_array_mml_integral(uli lo_sz = 1, uli hi_sz = 5,
                                         T lo_v = 1, T hi_v = 10) const;

  /**
   * @brief Creates a random array with real values.
   * @param shape The shape of the tensor to create.
   * @param lo_sz The lower bound of the array size.
   * @param hi_sz The upper bound of the array size.
   * @param lo_v The lower bound of the random values.
   * @param hi_v The upper bound of the random values.
   * @return A tensor with the specified shape and data. */
  array_mml<T> random_array_mml_real(uli lo_sz = 1, uli hi_sz = 5, T lo_v = 1,
                                     T hi_v = 100) const;

  void set_tensor_constructor(string id, function<void()> constructor);

private:
  TensorFactory() {
    this->tensor_constructor_1 = Tensor_mml::Tensor_mml;
    this->tensor_constructor_2 = Tensor_mml::Tensor_mml;
    this->tensor_constructor_3 = Tensor_mml::Tensor_mml;
    this->tensor_construcotr_3 = Tensor_mml::Tensor_mml;
  }

  void (*tensor_constructor_1)(const array_mml<uli> &shape,
                               const array_mml<T> &data);
  void (*tensor_constructor_2)(const array_mml<uli> &shape);
  void (*tensor_constructor_3)(const initializer_list<uli> shape,
                               const initializer_list<T> data);
  void (*tensor_constructor_4)(const initializer_list<uli> shape);
};

#include "../tensor_factory.tpp"