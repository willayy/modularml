#pragma once

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"

/*!
 * @brief A Tensor<T> implementation using an underlying
 * fixed size 1D array with row-major offsets for
 * multi-dimensional indexing.
 * @tparam T The type of the data contained in the tensor.
 * Allows for arithmetic types.
 */
template <typename T>
class Tensor_mml : public Tensor<T> {
 public:
  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  explicit Tensor_mml(initializer_list<uli> shape);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  explicit Tensor_mml(const array_mml<uli>& shape);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(initializer_list<uli> shape, initializer_list<T> data);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(array_mml<uli>& shape, array_mml<T>& data);

  /// @brief Destructor for Tensor_mml class.
  ~Tensor_mml() = default;

  /// @brief Move constructor for Tensor_mml class.
  Tensor_mml(Tensor_mml&& other) noexcept;

  /// @brief Copy constructor for Tensor_mml class.
  Tensor_mml(const Tensor_mml& other);

  /// @brief Get the raw 1D data of the tensor.
  /// @return The data of the tensor.
  const array_mml<T>& get_data() const;

  /// @brief Copy-Assignment operator for Tensor_mml class.
  /// @param other The tensor to assign.
  /// @return The copied tensor.
  Tensor_mml& operator=(const Tensor_mml& other);

  /// @brief Move-Assignment operator for Tensor_mml class.
  /// @param other The tensor to assign.
  /// @return The moved tensor.
  Tensor_mml& operator=(Tensor_mml&& other) noexcept;

  /// Ovveridden methods from the base class
  Tensor<T>& operator=(const Tensor<T>& other) override;
  Tensor<T>& operator=(Tensor<T>&& other) noexcept override;
  string to_string() const override;
  shared_ptr<Tensor<T>> copy() const override;
  void flip(uli dim) override;
  shared_ptr<Tensor<T>> slice(initializer_list<uli> slice_indices) override;
  shared_ptr<Tensor<T>> slice(array_mml<uli>& slice_indices) override;
  void reshape(const array_mml<uli>& new_shape) override;
  void reshape(initializer_list<uli> new_shape) override;
  bool is_matrix() const override;
  bool matrix_match(const Tensor<T>& other) const override;
  bool operator==(const Tensor<T>& other) const override;
  bool operator!=(const Tensor<T>& other) const override;
  const array_mml<uli>& get_shape() const override;
  const array_mml<uli>& get_offsets() const;
  uli get_size() const override;
  const T& operator[](array_mml<uli>& indices) const override;
  T& operator[](array_mml<uli>& indices) override;
  const T& operator[](initializer_list<uli> indices) const override;
  T& operator[](initializer_list<uli> indices) override;
  const T& operator[](uli index) const override;
  T& operator[](uli index) override;
  void fill(T value) override;

 private:
  array_mml<T> data;
  array_mml<uli> shape;
  array_mml<uli> offsets;
  uli size;

  // Helper methods
  array_mml<uli> compute_offsets() const;
  uli compute_size() const;
  bool valid_shape(const array_mml<uli>& new_shape) const;
  bool valid_indices(const array_mml<uli>& indices) const;
  bool valid_slice_indices(const array_mml<uli>& slice_indices) const;
  uli index_with_offset(array_mml<uli> indices) const;
};

// Convenience initializers
template <typename T>
Tensor<T> tensor_mml(const initializer_list<uli> shape);

template <typename T>
Tensor_mml<T> tensor_mml(const initializer_list<uli> shape, const initializer_list<T> data);

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<uli> shape);

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<uli> shape, const initializer_list<T> data);

/*  We include the implementation of the template class here
 *   because the compiler needs to see the implementation
 *   when instantiating the template with a specific type.
 *   This is a common but hacky practice when working with templates.  */
#include "../mml_tensor.tpp"