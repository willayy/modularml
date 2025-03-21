#pragma once

#include "mml_tensor.hpp"

template <typename T>
Tensor_mml<T>::Tensor_mml(
    initializer_list<uli> shape,
    optional<array_mml<uli>> slice_offsets) : Tensor<T>(),
                                              shape(shape) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets = has_value ? optional<array_mml<uli>>(slice_offsets.value()) : nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->size);
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(
    initializer_list<uli> shape,
    initializer_list<T> data,
    optional<array_mml<uli>> slice_offsets) : Tensor<T>(),
                                              shape(shape),
                                              data(data) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets = has_value ? optional<array_mml<uli>>(slice_offsets.value()) : nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
}

template <typename T>
Tensor_mml<T>::Tensor_mml(
    const array_mml<uli>& shape,
    optional<array_mml<uli>> slice_offsets) : Tensor<T>(),
                                              shape(shape) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets = has_value ? optional<array_mml<uli>>(slice_offsets.value()) : nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->size);
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(
    array_mml<uli>& shape,
    array_mml<T>& data,
    optional<array_mml<uli>> slice_offsets) : Tensor<T>(),
                                              shape(shape),
                                              data(data) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets = has_value ? optional<array_mml<uli>>(slice_offsets.value()) : nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
}

template <typename T>
Tensor_mml<T>::Tensor_mml(Tensor_mml&& other) noexcept : Tensor<T>(other) {
  this->shape = move(other.shape);
  this->indices_offsets = move(other.indices_offsets);
  this->size = other.size;

  bool has_value = other.slice_offsets.has_value();
  this->slice_offsets = has_value ? optional<array_mml<uli>>(move(other.slice_offsets.value())) : nullopt;
  this->data = move(other.data);
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const Tensor_mml& other) : Tensor<T>(other) {
  this->shape = array_mml<uli>(other.shape);
  this->indices_offsets = array_mml<uli>(other.indices_offsets);
  this->data = array_mml<T>(other.data);
  this->size = other.size;

  bool has_value = other.slice_offsets.has_value();
  this->slice_offsets = has_value ? optional<array_mml<uli>>(other.slice_offsets.value()) : nullopt;
}

template <typename T>
const array_mml<T>& Tensor_mml<T>::get_data() const {
  return this->data;
}

template <typename T>
Tensor<T>& Tensor_mml<T>::operator=(const Tensor<T>& other) {
  if (this != &other) {
    auto other_cast = dynamic_cast<const Tensor_mml<T>&>(other);
    this->shape = array_mml<uli>(other_cast.shape);
    this->indices_offsets = array_mml<uli>(other_cast.indices_offsets);
    this->data = array_mml<T>(other_cast.data);
    this->size = other_cast.size;

    bool has_value = other_cast.slice_offsets.has_value();
    this->slice_offsets = has_value ? optional<array_mml<uli>>(other_cast.slice_offsets.value()) : nullopt;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor_mml<T>::operator=(Tensor<T>&& other) noexcept {
  if (this != &other) {
    auto other_cast = dynamic_cast<Tensor_mml<T>&&>(move(other));
    this->data = move(other_cast.data);
    this->shape = move(other_cast.shape);
    this->indices_offsets = move(other_cast.indices_offsets);
    this->size = other_cast.size;

    bool has_value = other_cast.slice_offsets.has_value();
    this->slice_offsets = has_value ? optional<array_mml<uli>>(move(other_cast.slice_offsets.value())) : nullopt;
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
    string first_10 = "[";
    string last_10 = "";
    for (uli i = 0; i < 9; i++) {
      first_10 += std::to_string((*this)[i]) + ", ";
      last_10 += std::to_string((*this)[this->size - i - 1]) + ", ";
    }
    first_10 += std::to_string((*this)[9]) + ", ... ";
    last_10 += std::to_string((*this)[this->size - 10]) + "]";

    data_str += first_10 + " ... " + last_10;
  } else {
    data_str += "[";
    for (uli i = 0; i < this->size - 1; i++) {
      data_str += std::to_string((*this)[i]) + ", ";
    }
    data_str += std::to_string((*this)[this->size - 1]) + "]";
  }
  return base + ptr_str + ", " + shape_str + ", " + size_str + ", " + data_str;
}

template <typename T>
shared_ptr<Tensor<T>> Tensor_mml<T>::copy() const {
  return make_shared<Tensor_mml<T>>(*this);
}

template <typename T>
void Tensor_mml<T>::reshape(const array_mml<uli>& new_shape) {
  if (!valid_shape(new_shape)) throw invalid_argument("Invalid shape");
  this->shape = array_mml<uli>(new_shape);
  this->indices_offsets = compute_indices_offsets();
}

template <typename T>
void Tensor_mml<T>::reshape(initializer_list<uli> new_shape) {
  reshape(array_mml<uli>(new_shape));
}

template <typename T>
void Tensor_mml<T>::flip(uli dim) {
  if (!valid_flip_dimension(dim)) throw invalid_argument("Invalid flip dimension");
  for (uli i = 0; i < this->shape[dim]; i++) {
    array_mml<uli> indices = array_mml<uli>(dim + 1);
    indices.fill(0);
    indices[dim] = i;
    shared_ptr<Tensor<T>> slice = this->slice(indices);
    slice->reverse_buffer();
  }
}

template <typename T>
void Tensor_mml<T>::reverse_buffer() {
  uli i = 0;
  uli j = this->size - 1;
  while (i < j) {
    T temp = this->data[i];
    this->data[i] = this->data[j];
    this->data[j] = temp;
    i++;
    j--;
  }
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

  // Calculate and create the new shape for the sliced tensor.
  uli shape_size = this->shape.size();
  uli slice_indices_size = slice_indices.size();
  uli slice_shape_dif = shape_size - slice_indices_size;
  array_mml<uli> slice_shape = array_mml<uli>(slice_shape_dif);

  uli i = 0;
  uli j = shape_size - slice_shape_dif;

  if (slice_shape.size() == 1) {
    slice_shape[0] = this->shape[shape_size - 2];
  } else {
    while (i < slice_shape_dif) {
      slice_shape[i] = this->shape[j];
      i++;
      j++;
    }
  }

  // Create a shallow copy of the data that the sliced tensor will use.
  shared_ptr<T[]> data_ptr(this->data.get(), [](T*) { /* noop */ });
  array_mml<T> data_shallow_copy = array_mml<T>(data_ptr, this->data.size());
  // Calculate the new indices offsets for the sliced tensor.
  array_mml<uli> new_slice_offsets = compute_slice_offsets(slice_indices, slice_shape);

  // Return a new Tensor but with a shallow copy of the data.
  shared_ptr<Tensor<T>> sliced_tensor = make_shared<Tensor_mml<T>>(
      slice_shape,
      data_shallow_copy,
      new_slice_offsets);

  return sliced_tensor;
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
  if (this->get_size() != other.get_size()) return false;
  if (this->get_shape() != other.get_shape()) return false;

  for (uli i = 0; i < this->size; i++) {
    if ((*this)[i] != other[i]) {
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
  return this->indices_offsets;
}

template <typename T>
uli Tensor_mml<T>::get_size() const {
  return this->size;
}

template <typename T>
const T& Tensor_mml<T>::operator[](array_mml<uli>& indices) const {
  if (!valid_indices(indices)) throw invalid_argument("Invalid Tensor indices");
  if (this->slice_offsets.has_value()) return this->data[index_to_slice_index(indices_to_1d_index(indices))];
  return this->data[indices_to_1d_index(indices)];
}

template <typename T>
T& Tensor_mml<T>::operator[](array_mml<uli>& indices) {
  if (!valid_indices(indices)) throw invalid_argument("Invalid Tensor indices");
  if (this->slice_offsets.has_value()) return this->data[index_to_slice_index(indices_to_1d_index(indices))];
  return this->data[indices_to_1d_index(indices)];
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
  if (!valid_index(index)) throw invalid_argument("Invalid Tensor index");
  if (this->slice_offsets.has_value()) return this->data[index_to_slice_index(index)];
  return this->data[index];
}

template <typename T>
T& Tensor_mml<T>::operator[](uli index) {
  if (!valid_index(index)) throw invalid_argument("Invalid Tensor index");
  if (this->slice_offsets.has_value()) return this->data[index_to_slice_index(index)];
  return this->data[index];
}

template <typename T>
void Tensor_mml<T>::fill(T value) {
  this->data.fill(value);
}

template <typename T>
array_mml<uli> Tensor_mml<T>::compute_indices_offsets() const {
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
array_mml<uli> Tensor_mml<T>::compute_slice_offsets(array_mml<uli>& slice_indices, array_mml<uli>& slice_shape) const {
  uli jump_rows = 0;
  uli i_stride = 1;
  uli i_add = 0;

  if (slice_shape.size() % 2 != 0) {
    i_stride = this->shape[ this->shape.size() - 1 ];
    i_add = slice_indices[ slice_indices.size() - 1 ];
  }
  
  // Special rule for slicing every tensor that doesnt have shape {2, 2}
  if (this->shape.size() != 2) {
    jump_rows = slice_indices[0] * this->indices_offsets[0];
  }

  if (this->slice_offsets.has_value()) {
    jump_rows += this->slice_offsets.value()[0];
  }

  return array_mml<uli>({jump_rows, i_stride, i_add});
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
bool Tensor_mml<T>::valid_index(uli index) const {
  return index < this->get_size();
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
uli Tensor_mml<T>::indices_to_1d_index(array_mml<uli> indices) const {
  auto index = 0;
  for (uli i = 0; i < indices.size(); i++) {
    index += (indices[i]) * this->indices_offsets[i];
  }
  return index;
}

template <typename T>
uli Tensor_mml<T>::index_to_slice_index(uli index) const {
  if (!this->slice_offsets.has_value()) throw invalid_argument("No slice offsets available");

  uli jump_rows = this->slice_offsets.value()[0]; // How many rows should we jump in 1d buffer
  uli i_stride = this->slice_offsets.value()[1]; // How big is the stride per i index
  uli i_add = this->slice_offsets.value()[2]; // How much should we add to i to offset it

  uli slice_index = jump_rows + index * i_stride + i_add;
  return slice_index;
}

template <typename T>
bool Tensor_mml<T>::valid_slice_indices(const array_mml<uli>& slice_indices) const {
  if (slice_indices.size() >= this->shape.size()) {
    return false;
  }
  uli slice_shape_dif = this->shape.size() - slice_indices.size();
  for (uli i = 0; i < slice_indices.size(); i++) {
    if (slice_indices[i] >= this->shape[i + slice_shape_dif]) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Tensor_mml<T>::valid_flip_dimension(uli dim) const {
  return dim < this->shape.size();
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