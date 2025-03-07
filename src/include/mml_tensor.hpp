#pragma once

#include "a_tensor.hpp"
#include "array_mml.hpp"
#include "globals.hpp"

/// @brief
/// @tparam T
template <typename T>
class Tensor_mml : public Tensor<T> {
 public:
  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  explicit Tensor_mml(initializer_list<int> shape) : Tensor<T>(shape) {
    this->data = array_mml<T>(this->get_size());
    this->data.fill(T(0));
  }

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  explicit Tensor_mml(array_mml<int>& shape) : Tensor<T>(shape) {
    this->data = array_mml<T>(this->get_size());
    this->data.fill(T(0));
  }

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(initializer_list<int> shape, initializer_list<T> data) : Tensor<T>(shape) {
    this->data = array_mml<T>(data);
  }

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(array_mml<int>& shape, array_mml<T>& data) : Tensor<T>(shape) {
    this->data = array_mml<T>(data);
  }

  /// @brief Destructor for Tensor_mml class.
  ~Tensor_mml() = default;

  /// @brief Move constructor for Tensor_mml class.
  Tensor_mml(Tensor_mml&& other) noexcept : Tensor<T>(move(other)), data(move(other.data)) {}

  /// @brief Copy constructor for Tensor_mml class.
  Tensor_mml(const Tensor_mml& other) : Tensor<T>(other), data(array_mml<T>(other.data)) {}

  /// @brief Move assignment operator.
  /// @param other The tensor to move.
  /// @return The moved tensor.
  Tensor_mml& operator=(Tensor_mml&& other) noexcept {
    Tensor<T>::operator=(move(other));
    this->data = move(other.data);
    return *this;
  }

  /// @brief Get the data of the tensor.
  /// @return The data of the tensor.
  const array_mml<T>& get_data() const {
    return this->data;
  }

  /// @brief Assignment operator.
  ConcreteTensor& operator=(const AbstractTensor& other) override {
    // Make sure the other tensor is of the same type because data is of type ConcreteTensor
    const ConcreteTensor<T>& otherTensor = static_cast<const ConcreteTensor<T>&>(other);

    // Copy base members
    this->shape = otherTensor.shape;
    this->offsets = otherTensor.offsets;
    this->size = otherTensor.size;

    // Copy the data array
    this->data = otherTensor.data;
    return *this;
  }

  string to_string() const override {
    string base = Tensor<T>::to_string();
    return base + " Data: " + this->data.to_string();
    return this->data.to_string();
  }

  shared_ptr<Tensor<T>> copy() const override {
    return make_shared<Tensor_mml<T>>(*this);
  }

  const T& operator[](int index) const override {
    return this->data[index];
  }

  T& operator[](int index) override {
    return this->data[index];
  }

 private:
  array_mml<T> data;
};

// Convience initializers

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new tensor with the given shape and all elements set to zero.
template <typename T>
[[deprecated("Use Tensor_mml constructor instead.")]]
Tensor<T> tensor_mml(const initializer_list<int> shape) {  // NOSONAR - function signature is correct
  auto t = make_shared<Tensor_mml<T>>(shape);
  return t;
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new tensor with the given shape and data.
template <typename T>
[[deprecated("Use Tensor_mml constructor instead.")]]
Tensor_mml<T> tensor_mml(const initializer_list<int> shape, const initializer_list<T> data) {  // NOSONAR - function signature is correct
  auto t = Tensor_mml<T>(shape, data);
  return t;
}

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new shared tensor pointer with the given shape and all elements set to zero.
template <typename T>
[[deprecated("Use Tensor_mml constructor instead.")]]
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape) {  // NOSONAR - function signature is correct
  auto t = make_shared<Tensor_mml<T>>(shape);
  return t;
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new shared tensor pointer with the given shape and data.
template <typename T>
[[deprecated("Use Tensor_mml constructor instead.")]]
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape, const initializer_list<T> data) {  // NOSONAR - function signature is correct
  auto t = make_shared<Tensor_mml<T>>(shape, data);
  return t;
}