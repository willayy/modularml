#pragma once

#include "mml_tensor.hpp"

template <typename T>
Tensor_mml<T>::Tensor_mml(initializer_list<uli> shape)
    : Tensor<T>(),
      shape(shape) {
  this->offsets = compute_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->get_size());
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const array_mml<uli>& shape)
    : Tensor<T>(),
      shape(shape) {
  this->offsets = compute_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->get_size());
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(initializer_list<uli> shape, initializer_list<T> data)
    : Tensor<T>(),
      shape(shape),
      data(data) {
  this->offsets = compute_offsets();
  this->size = compute_size();
}

template <typename T>
Tensor_mml<T>::Tensor_mml(array_mml<uli>& shape, array_mml<T>& data)
    : Tensor<T>(),
      shape(shape),
      data(data) {
  this->offsets = compute_offsets();
  this->size = compute_size();
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
  this->shape = array_mml<uli>(other.shape);
  this->offsets = array_mml<uli>(other.offsets);
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
    this->shape = array_mml<uli>(other.shape);
    this->offsets = array_mml<uli>(other.offsets);
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
    this->shape = array_mml<uli>(other_cast.shape);
    this->offsets = array_mml<uli>(other_cast.offsets);
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
  string ptr_str = "Pointer: " + std::to_string(reinterpret_cast<uli>(this));
  string shape_str = "Shape: " + this->shape.to_string();
  string size_str = "Size: " + std::to_string(this->size);
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
void Tensor_mml<T>::reshape(const array_mml<uli>& new_shape) {
  if (!valid_shape(new_shape)) {
    throw invalid_argument("Invalid shape");
  }
  this->shape = array_mml<uli>(new_shape);
  this->offsets = compute_offsets();
}

template <typename T>
void Tensor_mml<T>::reshape(initializer_list<uli> new_shape) {
  reshape(array_mml<uli>(new_shape));
}

template <typename T>
void Tensor_mml<T>::flip(uli dim) {

}

template <typename T>
shared_ptr<Tensor<T>> Tensor_mml<T>::slice(initializer_list<uli> slice_indices) {
  auto slice_indices_array = array_mml<uli>(slice_indices);
  return slice(slice_indices_array);
}

template <typename T>
shared_ptr<Tensor<T>> Tensor_mml<T>::slice(array_mml<uli>& slice_indices) {
  if (!valid_slice_indices(slice_indices)) {
    throw invalid_argument("Invalid slice indices");
  }

  uli slice_shape_dif = this->shape.size() - slice_indices.size();
  array_mml<uli> slice_shape = array_mml<uli>(slice_shape_dif);

  for (uli i = 0; i < slice_shape_dif; i++) {
    slice_shape[i] = this->shape[i + 1];
  }

  uli data_slice_start = index_with_offset(slice_indices);
  uli data_slice_size = accumulate(slice_shape.begin(), slice_shape.end(), 1, multiplies<uli>());
  array_mml<T> data_slice = this->data.m_subarray(data_slice_start, data_slice_start + data_slice_size);

  return make_shared<Tensor_mml<T>>(slice_shape, data_slice);
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
  for (uli i = 0; i < this->get_size(); i++) {
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
const array_mml<uli>& Tensor_mml<T>::get_shape() const {
  return this->shape;
}

template <typename T>
const array_mml<uli>& Tensor_mml<T>::get_offsets() const {
  return this->offsets;
}

template <typename T>
uli Tensor_mml<T>::get_size() const {
  return this->size;
}

template <typename T>
const T& Tensor_mml<T>::operator[](array_mml<uli>& indices) const {
  if (!valid_indices(indices)) {
    throw invalid_argument("Invalid indices");
  }
  return this->data[index_with_offset(indices)];
}

template <typename T>
T& Tensor_mml<T>::operator[](array_mml<uli>& indices) {
  if (!valid_indices(indices)) {
    throw invalid_argument("Invalid indices");
  }
  return this->data[index_with_offset(indices)];
}

template <typename T>
const T& Tensor_mml<T>::operator[](initializer_list<uli> indices) const {
  auto indices_array = array_mml<uli>(indices);
  return (*this)[indices_array];
}

template <typename T>
T& Tensor_mml<T>::operator[](initializer_list<uli> indices) {
  auto indices_array = array_mml<uli>(indices);
  return (*this)[indices_array];
}

template <typename T>
const T& Tensor_mml<T>::operator[](uli index) const {
  if (index >= this->get_size()) {
    throw invalid_argument("Invalid index");
  }
  return this->data[index];
}

template <typename T>
T& Tensor_mml<T>::operator[](uli index) {
  if (index >= this->get_size()) {
    throw invalid_argument("Invalid index");
  }
  return this->data[index];
}

template <typename T>
void Tensor_mml<T>::fill(T value) {
  this->data.fill(value);
}

template <typename T>
array_mml<uli> Tensor_mml<T>::compute_offsets() const {
  const uli shape_size = this->shape.size();
  auto computed_offsets = array_mml<uli>(shape_size);
  computed_offsets.fill(1);
  // Special case if shape is 1D
  if (shape_size == 1) {
    return computed_offsets;
  }
  // Compute offsets
  uli i = shape_size - 2;
  do {
    computed_offsets[i] = this->shape[i + 1] * computed_offsets[i + 1];
  } while (i-- > 0);
  return computed_offsets;
}

template <typename T>
uli Tensor_mml<T>::compute_size() const {
  return accumulate(this->shape.begin(), this->shape.end(), 1, multiplies<uli>());
}

template <typename T>
bool Tensor_mml<T>::valid_shape(const array_mml<uli>& new_shape) const {
  return accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<uli>()) == this->get_size();
}

template <typename T>
bool Tensor_mml<T>::valid_indices(const array_mml<uli>& indices) const {
  if (indices.size() != this->shape.size()) {
    return false;
  }
  for (uli i = 0; i < indices.size(); i++) {
    if (indices[i] >= this->shape[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
uli Tensor_mml<T>::index_with_offset(array_mml<uli> indices) const {
  auto index = 0;
  for (uli i = 0; i < indices.size(); i++) {
    index += (indices[i]) * this->offsets[i];
  }
  return index;
}

template <typename T>
bool Tensor_mml<T>::valid_slice_indices(const array_mml<uli>& slice_indices) const {
  if (slice_indices.size() >= this->shape.size()) {
    return false;
  }
  for (uli i = 0; i < slice_indices.size(); i++) {
    if (slice_indices[i] >= this->shape[i]) {
      return false;
    }
  }
  return true;
}

// Convenience initializers

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<uli> shape) {
  auto t = make_shared<Tensor_mml<T>>(shape);
  return t;
}

template <typename T>
shared_ptr<Tensor<T>> tensor_mml_p(const initializer_list<uli> shape, const initializer_list<T> data) {
  auto t = make_shared<Tensor_mml<T>>(shape, data);
  return t;
}