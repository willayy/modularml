#pragma once

#include "mml_tensor.hpp"

template <typename T>
Tensor_mml<T>::Tensor_mml(initializer_list<int> shape) : Tensor<T>() {
  this->shape = array_mml<int>(shape);
  this->offsets = compute_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->get_size());
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const array_mml<int>& shape) : Tensor<T>() {
  this->shape = array_mml<int>(shape);
  this->offsets = compute_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->get_size());
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(initializer_list<int> shape, initializer_list<T> data) : Tensor<T>() {
  this->shape = array_mml<int>(shape);
  this->offsets = compute_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(data);
}

template <typename T>
Tensor_mml<T>::Tensor_mml(array_mml<int>& shape, array_mml<T>& data) : Tensor<T>() {
  this->shape = array_mml<int>(shape);
  this->offsets = compute_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(data);
}

template <typename T>
Tensor_mml<T>::Tensor_mml(Tensor_mml&& other) noexcept : Tensor<T>(other) {
  this->shape = move(other.shape);
  this->offsets = move(other.offsets);
  this->size = other.size;
  this->data = move(other.data);
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const Tensor_mml& other) : Tensor<T>(other) {
  this->shape = array_mml<int>(other.shape);
  this->offsets = array_mml<int>(other.offsets);
  this->data = array_mml<T>(other.data);
  this->size = other.size;
}

template <typename T>
const array_mml<T>& Tensor_mml<T>::get_data() const {
  return this->data;
}

template <typename T>
Tensor_mml<T>& Tensor_mml<T>::operator=(const Tensor_mml& other) {
  if (this != &other) {
    this->shape = array_mml<int>(other.shape);
    this->offsets = array_mml<int>(other.offsets);
    this->size = other.size;
    this->data = array_mml<T>(other.data);
  }
  return *this;
}

template <typename T>
Tensor_mml<T>& Tensor_mml<T>::operator=(Tensor_mml&& other) noexcept {
  if (this != &other) {
    this->shape = move(other.shape);
    this->offsets = move(other.offsets);
    this->size = other.size;
    this->data = move(other.data);
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor_mml<T>::operator=(const Tensor<T>& other) {
  if (this != &other) {
    auto other_cast = dynamic_cast<const Tensor_mml<T>&>(other);
    this->data = array_mml<T>(other_cast.data);
    this->shape = array_mml<int>(other_cast.shape);
    this->offsets = array_mml<int>(other_cast.offsets);
    this->size = other_cast.size;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor_mml<T>::operator=(Tensor<T>&& other) noexcept {
  if (this != &other) {
    auto other_cast = dynamic_cast<Tensor_mml<T>&&>(move(other));
    this->data = move(other_cast.data);
    this->shape = move(other_cast.shape);
    this->offsets = move(other_cast.offsets);
    this->size = other_cast.size;
  }
  return *this;
}

template <typename T>
string Tensor_mml<T>::to_string() const {
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

template <typename T>
shared_ptr<Tensor<T>> Tensor_mml<T>::copy() const {
  return make_shared<Tensor_mml<T>>(*this);
}

template <typename T>
void Tensor_mml<T>::reshape(const array_mml<int>& new_shape) {
  if (!valid_shape(new_shape)) {
    throw invalid_argument("Invalid shape");
  }
  this->shape = array_mml<int>(new_shape);
  this->offsets = compute_offsets();
}

template <typename T>
void Tensor_mml<T>::reshape(initializer_list<int> new_shape) {
  reshape(array_mml<int>(new_shape));
}

template <typename T>
bool Tensor_mml<T>::is_matrix() const {
  return this->shape.size() == 2;
}

template <typename T>
bool Tensor_mml<T>::matrix_match(const Tensor<T>& other) const {
  if (!this->is_matrix() || !other.is_matrix()) {
    return false;
  }
  return this->get_shape()[1] == other.get_shape()[0];
}

template <typename T>
bool Tensor_mml<T>::operator==(const Tensor<T>& other) const {
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

template <typename T>
bool Tensor_mml<T>::operator!=(const Tensor<T>& other) const {
  return !(*this == other);
}

template <typename T>
const array_mml<int>& Tensor_mml<T>::get_shape() const {
  return this->shape;
}

template <typename T>
const array_mml<int>& Tensor_mml<T>::get_offsets() const {
  return this->offsets;
}

template <typename T>
uint64_t Tensor_mml<T>::get_size() const {
  return this->size;
}

template <typename T>
const T& Tensor_mml<T>::operator[](array_mml<int>& indices) const {
  if (!valid_indices(indices)) {
    throw invalid_argument("Invalid indices");
  }
  return this->data[index_with_offset(indices)];
}

template <typename T>
T& Tensor_mml<T>::operator[](array_mml<int>& indices) {
  if (!valid_indices(indices)) {
    throw invalid_argument("Invalid indices");
  }
  return this->data[index_with_offset(indices)];
}

template <typename T>
const T& Tensor_mml<T>::operator[](initializer_list<int> indices) const {
  auto indices_array = array_mml<int>(indices);
  return (*this)[indices_array];
}

template <typename T>
T& Tensor_mml<T>::operator[](initializer_list<int> indices) {
  auto indices_array = array_mml<int>(indices);
  return (*this)[indices_array];
}

template <typename T>
const T& Tensor_mml<T>::operator[](int index) const {
  if (index < 0 || index >= this->get_size()) {
    throw invalid_argument("Invalid index");
  }
  return this->data[index];
}

template <typename T>
T& Tensor_mml<T>::operator[](int index) {
  if (index < 0 || index >= this->get_size()) {
    throw invalid_argument("Invalid index");
  }
  return this->data[index];
}

template <typename T>
void Tensor_mml<T>::fill(T value) {
  this->data.fill(value);
}

template <typename T>
array_mml<int> Tensor_mml<T>::compute_offsets() const {
  const int shape_size = static_cast<int>(shape.size());
  auto computed_offsets = array_mml<int>(shape_size);
  computed_offsets.fill(1);
  for (int i = shape_size - 2; i >= 0; i--) {
    computed_offsets[i] = this->shape[i + 1] * computed_offsets[i + 1];
  }
  return computed_offsets;
}

template <typename T>
uint64_t Tensor_mml<T>::compute_size() const {
  return accumulate(this->shape.begin(), this->shape.end(), 1, multiplies<int>());
}

template <typename T>
bool Tensor_mml<T>::valid_shape(const array_mml<int>& new_shape) const {
  return accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>()) == this->get_size();
}

template <typename T>
bool Tensor_mml<T>::valid_indices(const array_mml<int>& indices) const {
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

template <typename T>
int Tensor_mml<T>::index_with_offset(array_mml<int> indices) const {
  auto index = 0;
  const auto shape_size = static_cast<int>(shape.size());
  for (int i = 0; i < shape_size; i++) {
    index += (indices[i]) * this->offsets[i];
  }
  return index;
}

// Convenience initializers
template <typename T>
Tensor<T> tensor_mml(const initializer_list<int> shape) {
  auto t = make_shared<Tensor_mml<T>>(shape);
  return t;
}

template <typename T>
Tensor_mml<T> tensor_mml(const initializer_list<int> shape, const initializer_list<T> data) {
  auto t = Tensor_mml<T>(shape, data);
  return t;
}

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape) {
  auto t = make_shared<Tensor_mml<T>>(shape);
  return t;
}

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<int> shape, const initializer_list<T> data) {
  auto t = make_shared<Tensor_mml<T>>(shape, data);
  return t;
}

