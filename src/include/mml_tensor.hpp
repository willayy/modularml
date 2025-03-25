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
  explicit Tensor_mml(initializer_list<int> shape);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  explicit Tensor_mml(const array_mml<int>& shape);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(initializer_list<int> shape, initializer_list<T> data);

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(array_mml<int>& shape, array_mml<T>& data);

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
  void reshape(const array_mml<int>& new_shape) override;
  void reshape(initializer_list<int> new_shape) override;
  bool is_matrix() const override;
  bool matrix_match(const Tensor<T>& other) const override;
  bool operator==(const Tensor<T>& other) const override;
  bool operator!=(const Tensor<T>& other) const override;
  const array_mml<int>& get_shape() const override;
  const array_mml<int>& get_offsets() const;
  uint64_t get_size() const override;
  const T& operator[](array_mml<int>& indices) const override;
  T& operator[](array_mml<int>& indices) override;
  const T& operator[](initializer_list<int> indices) const override;
  T& operator[](initializer_list<int> indices) override;
  const T& operator[](int index) const override;
  T& operator[](int index) override;
  void fill(T value) override;

 private:
  array_mml<T> data;
  array_mml<int> shape;
  array_mml<int> offsets;
  uint64_t size;

  // Helper methods
  array_mml<int> compute_offsets() const;
  uint64_t compute_size() const;
  bool valid_shape(const array_mml<int>& new_shape) const;
  bool valid_indices(const array_mml<int>& indices) const;
  int index_with_offset(array_mml<int> indices) const;
};

// Convenience initializers
template <typename T>
Tensor<T> tensor_mml(const initializer_list<int> shape);

template <typename T>
Tensor_mml<T> tensor_mml(const initializer_list<int> shape, const initializer_list<T> data);

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape);

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape, const initializer_list<T> data);

/*  We include the implementation of the template class here
*   because the compiler needs to see the implementation
*   when instantiating the template with a specific type.
*   This is a common but hacky practice when working with templates.  */
#include "../mml_tensor.tpp"