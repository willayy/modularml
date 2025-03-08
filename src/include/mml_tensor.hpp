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
  explicit Tensor_mml(initializer_list<int> shape) : Tensor<T>() {
    this->shape = array_mml<int>(shape);
    this->offsets = compute_offsets();
    this->size = compute_size();
    this->data = array_mml<T>(this->get_size());
    this->data.fill(T(0));
  }

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  explicit Tensor_mml(const array_mml<int>& shape) : Tensor<T>() {
    this->shape = array_mml<int>(shape);
    this->offsets = compute_offsets();
    this->size = compute_size();
    this->data = array_mml<T>(this->get_size());
    this->data.fill(T(0));
  }

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(initializer_list<int> shape, initializer_list<T> data) : Tensor<T>() {
    this->shape = array_mml<int>(shape);
    this->offsets = compute_offsets();
    this->size = compute_size();
    this->data = array_mml<T>(data);
  }

  /// @brief Constructor for Tensor_mml class.
  /// @param shape The shape of the tensor.
  /// @param data The data to set in the tensor.
  explicit Tensor_mml(array_mml<int>& shape, array_mml<T>& data) : Tensor<T>() {
    this->shape = array_mml<int>(shape);
    this->offsets = compute_offsets();
    this->size = compute_size();
    this->data = array_mml<T>(data);
  }

  /// @brief Destructor for Tensor_mml class.
  ~Tensor_mml() = default;

  /// @brief Move constructor for Tensor_mml class.

  Tensor_mml(Tensor_mml&& other) noexcept : Tensor<T>(other) {
    this->shape = move(other.shape);
    this->offsets = move(other.offsets);
    this->size = other.size;
    this->data = move(other.data);
  }

  /// @brief Copy constructor for Tensor_mml class.
  Tensor_mml(const Tensor_mml& other) : Tensor<T>(other) {
    this->shape = array_mml<int>(other.shape);
    this->offsets = array_mml<int>(other.offsets);
    this->data = array_mml<T>(other.data);
    this->size = other.size;
  }

  /// @brief Get the data of the tensor.
  /// @return The data of the tensor.
  const array_mml<T>& get_data() const {
    return this->data;
  }

  /// @brief Copy-Assignment operator for Tensor_mml class.
  /// @param other The tensor to assign.
  /// @return The copied tensor.
  Tensor_mml& operator=(const Tensor_mml& other) {
    if (this != &other) {
      this->shape = array_mml<int>(other.shape);
      this->offsets = array_mml<int>(other.offsets);
      this->size = other.size;
      this->data = array_mml<T>(other.data);
    }
    return *this;
  }

  /// @brief Move-Assignment operator for Tensor_mml class.
  /// @param other The tensor to assign.
  /// @return The moved tensor.
  Tensor_mml& operator=(Tensor_mml&& other) noexcept {
    if (this != &other) {
      this->shape = move(other.shape);
      this->offsets = move(other.offsets);
      this->size = other.size;
      this->data = move(other.data);
    }
    return *this;
  }

  /// Ovveridden methods from the base class

  Tensor<T>& operator=(const Tensor<T>& other) override {
    if (this != &other) {
      auto other_cast = dynamic_cast<const Tensor_mml<T>&>(other);
      this->data = array_mml<T>(other_cast.data);
      this->shape = array_mml<int>(other_cast.shape);
      this->offsets = array_mml<int>(other_cast.offsets);
      this->size = other_cast.size;
    }
    return *this;
  }

  Tensor<T>& operator=(Tensor<T>&& other) noexcept override {
    if (this != &other) {
      auto other_cast = dynamic_cast<Tensor_mml<T>&&>(move(other));
      this->data = move(other_cast.data);
      this->shape = move(other_cast.shape);
      this->offsets = move(other_cast.offsets);
      this->size = other_cast.size;
    }
    return *this;
  }

  string to_string() const override {
    string base = string("Tensor_mml<") + typeid(T).name() + "> ";
    string ptr_str = "Pointer: " + std::to_string(reinterpret_cast<uintptr_t>(this));
    string shape_str = "Shape: " + this->shape.to_string();
    string size_str = "Size: " + std::to_string(int(this->size));
    string data_str = "Data: ";
    if (this->size > 30) {
      string first_10 = (this->data.subarray(0, 10)).to_string();
      string last_10 = (this->data.subarray(this->size - 10, 10)).to_string();
      data_str += first_10 + ", ..., " + last_10;
    } else {
      data_str += this->data.to_string();
    }
    return base + ptr_str + ", " + shape_str + ", " + size_str + ", " + data_str;
  }

  shared_ptr<Tensor<T>> copy() const override {
    return make_shared<Tensor_mml<T>>(*this);
  }

  void reshape(const array_mml<int>& new_shape) override {
    if (!valid_shape(new_shape)) {
      throw invalid_argument("Invalid shape");
    }
    this->shape = array_mml<int>(new_shape);
    this->offsets = compute_offsets();
  }

  void reshape(initializer_list<int> new_shape) override {
    reshape(array_mml<int>(new_shape));
  }

  bool is_matrix() const override {
    return this->shape.size() == 2;
  }

  bool matrix_match(const Tensor<T>& other) const override {
    if (!this->is_matrix() || !other.is_matrix()) {
      return false;
    }
    return this->get_shape()[1] == other.get_shape()[0];
  }

  bool operator==(const Tensor<T>& other) const override {
    if (this->get_size() != other.get_size()) {
      return false;
    }
    for (int i = 0; i < this->get_size(); i++) {
      if (this->data[i] != other[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Tensor<T>& other) const override {
    return !(*this == other);
  }

  const array_mml<int>& get_shape() const override {
    return this->shape;
  }

  const array_mml<int>& get_offsets() const {
    return this->offsets;
  }

  uint64_t get_size() const override {
    return this->size;
  }

  const T& operator[](array_mml<int>& indices) const override {
    if (!valid_indices(indices)) {
      throw invalid_argument("Invalid indices");
    }
    return this->data[index_with_offset(indices)];
  }

  T& operator[](array_mml<int>& indices) override {
    if (!valid_indices(indices)) {
      throw invalid_argument("Invalid indices");
    }
    return this->data[index_with_offset(indices)];
  }

  const T& operator[](initializer_list<int> indices) const override {
    auto indices_array = array_mml<int>(indices);
    return (*this)[indices_array];
  }

  T& operator[](initializer_list<int> indices) override {
    auto indices_array = array_mml<int>(indices);
    return (*this)[indices_array];
  }

  const T& operator[](int index) const override {
    if (index < 0 || index >= this->get_size()) {
      throw invalid_argument("Invalid index");
    }
    return this->data[index];
  }

  T& operator[](int index) override {
    if (index < 0 || index >= this->get_size()) {
      throw invalid_argument("Invalid index");
    }
    return this->data[index];
  }

  void fill(T value) override {
    this->data.fill(value);
  }

 private:
  array_mml<T> data;

  /// @brief The shape of the tensor.
  array_mml<int> shape;

  /// @brief The row-major offsets for the tensor.
  array_mml<int> offsets;

  /// @brief the size of the tensor.
  uint64_t size;

  // Helper methods

  /// @brief Row-major offsets for the data structure.
  /// @return a vector of integers representing the offsets.
  array_mml<int> compute_offsets() const {
    const int shape_size = static_cast<int>(shape.size());
    auto computed_offsets = array_mml<int>(shape_size);
    computed_offsets.fill(1);
    for (int i = shape_size - 2; i >= 0; i--) {
      computed_offsets[i] = this->shape[i + 1] * computed_offsets[i + 1];
    }
    return computed_offsets;
  }

  /// @brief Calculate the size of the tensor from the shape.
  /// @return The size of the tensor.
  uint64_t compute_size() const {
    return accumulate(this->shape.begin(), this->shape.end(), 1, multiplies<int>());
  }

  /// @brief Check if the new shape is valid.
  /// @param new_shape The new shape to check.
  /// @return True if the shape is valid, false otherwise.
  bool valid_shape(const array_mml<int>& new_shape) const {
    return accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>()) == this->get_size();
  }

  /// @brief Check if the indices are valid.
  /// @param indices The indices to check.
  /// @return True if the indices are valid, false otherwise.
  bool valid_indices(const array_mml<int>& indices) const {
    if (indices.size() != this->shape.size()) {
      return false;
    }
    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
      if (indices[i] < 0 || indices[i] >= this->shape[i]) {
        return false;
      }
    }
    return true;
  }

  /// @brief Calculates the 1_D index from the multi-dimensional indices.
  /// @param indices The indices to get the index for.
  /// @return The index.
  int index_with_offset(array_mml<int> indices) const {
    auto index = 0;
    const auto shape_size = static_cast<int>(shape.size());
    for (int i = 0; i < shape_size; i++) {
      index += (indices[i]) * this->offsets[i];
    }
    return index;
  }
};

// Convience initializers

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new tensor with the given shape and all elements set to zero.
template <typename T>
Tensor<T> tensor_mml(const initializer_list<int> shape) {  // NOSONAR - function signature is correct
  auto t = make_shared<Tensor_mml<T>>(shape);
  return t;
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new tensor with the given shape and data.
template <typename T>
Tensor_mml<T> tensor_mml(const initializer_list<int> shape, const initializer_list<T> data) {  // NOSONAR - function signature is correct
  auto t = Tensor_mml<T>(shape, data);
  return t;
}

/// @brief Initializes a new tensor with the given shape and all elements set to zero.
/// @param shape The shape of the tensor.
/// @return A new shared tensor pointer with the given shape and all elements set to zero.
template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape) {  // NOSONAR - function signature is correct
  auto t = make_shared<Tensor_mml<T>>(shape);
  return t;
}

/// @brief Initializes a new tensor with the given shape and data.
/// @param shape The shape of the tensor.
/// @param data A reference to the data to be set in the tensor.
/// @return A new shared tensor pointer with the given shape and data.
template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape, const initializer_list<T> data) {  // NOSONAR - function signature is correct
  auto t = make_shared<Tensor_mml<T>>(shape, data);
  return t;
}