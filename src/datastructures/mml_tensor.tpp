#pragma once

#include "datastructures/mml_tensor.hpp"

template <typename T>
Tensor_mml<T>::Tensor_mml(const std::initializer_list<uli> shape,
                          std::optional<array_mml<uli>> slice_offsets)
    : Tensor<T>(), shape(shape) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets =
      has_value ? std::optional<array_mml<uli>>(slice_offsets.value())
                : std::nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->size);
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const std::initializer_list<uli> shape,
                          const std::initializer_list<T> data,
                          std::optional<array_mml<uli>> slice_offsets)
    : Tensor<T>(), shape(shape), data(data) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets =
      has_value ? std::optional<array_mml<uli>>(slice_offsets.value())
                : std::nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const array_mml<uli> &shape,
                          std::optional<array_mml<uli>> slice_offsets)
    : Tensor<T>(), shape(shape) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets =
      has_value ? std::optional<array_mml<uli>>(slice_offsets.value())
                : std::nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->size);
  this->data.fill(T(0));
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const array_mml<uli> &shape, const array_mml<T> &data,
                          std::optional<array_mml<uli>> slice_offsets)
    : Tensor<T>(), shape(shape), data(data) {
  bool has_value = slice_offsets.has_value();
  this->slice_offsets =
      has_value ? std::optional<array_mml<uli>>(slice_offsets.value())
                : std::nullopt;
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
}

template <typename T>
Tensor_mml<T>::Tensor_mml(Tensor_mml &&other) noexcept : Tensor<T>(other) {
  this->shape = std::move(other.shape);
  this->indices_offsets = std::move(other.indices_offsets);
  this->size = other.size;

  bool has_value = other.slice_offsets.has_value();
  this->slice_offsets = has_value ? std::optional<array_mml<uli>>(
                                        std::move(other.slice_offsets.value()))
                                  : std::nullopt;
  this->data = std::move(other.data);
}

template <typename T>
Tensor_mml<T>::Tensor_mml(const Tensor_mml &other) : Tensor<T>(other) {
  this->shape = array_mml<uli>(other.shape);
  this->indices_offsets = array_mml<uli>(other.indices_offsets);
  this->data = array_mml<T>(other.data);
  this->size = other.size;

  bool has_value = other.slice_offsets.has_value();
  this->slice_offsets =
      has_value ? std::optional<array_mml<uli>>(other.slice_offsets.value())
                : std::nullopt;
}

template <typename T> const array_mml<T> &Tensor_mml<T>::get_data() const {
  return this->data;
}

template <typename T>
Tensor<T> &Tensor_mml<T>::operator=(const Tensor<T> &other) {
  if (this != &other) {
    auto other_cast = dynamic_cast<const Tensor_mml<T> &>(other);
    this->shape = array_mml<uli>(other_cast.shape);
    this->indices_offsets = array_mml<uli>(other_cast.indices_offsets);
    this->data = array_mml<T>(other_cast.data);
    this->size = other_cast.size;

    bool has_value = other_cast.slice_offsets.has_value();
    this->slice_offsets =
        has_value
            ? std::optional<array_mml<uli>>(other_cast.slice_offsets.value())
            : std::nullopt;
  }
  return *this;
}

template <typename T>
Tensor<T> &Tensor_mml<T>::operator=(Tensor<T> &&other) noexcept {
  if (this != &other) {
    auto other_cast = dynamic_cast<Tensor_mml<T> &&>(std::move(other));
    this->data = std::move(other_cast.data);
    this->shape = std::move(other_cast.shape);
    this->indices_offsets = std::move(other_cast.indices_offsets);
    this->size = other_cast.size;

    bool has_value = other_cast.slice_offsets.has_value();
    this->slice_offsets = has_value ? std::optional<array_mml<uli>>(std::move(
                                          other_cast.slice_offsets.value()))
                                    : std::nullopt;
  }
  return *this;
}

template <typename T> std::string Tensor_mml<T>::to_string() const {
  std::string base = std::string("Tensor_mml<") + typeid(T).name() + "> ";
  std::string ptr_str =
      "Pointer: " + std::to_string(reinterpret_cast<uintptr_t>(this));
  std::string shape_str = "Shape: " + this->shape.to_string();
  std::string size_str = "Size: " + std::to_string(this->size);
  std::string data_str = "Data: ";
  if (this->size > 30) {
    std::string first_10 = "[";
    std::string last_10 = "";
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
std::shared_ptr<Tensor<T>> Tensor_mml<T>::copy() const {
  return std::make_shared<Tensor_mml<T>>(*this);
}

template <typename T>
void Tensor_mml<T>::reshape(const array_mml<uli> &new_shape) {
  if (!valid_shape(new_shape))
    throw std::invalid_argument("Invalid shape");
  this->shape = array_mml<uli>(new_shape);
  this->indices_offsets = compute_indices_offsets();
}

template <typename T>
void Tensor_mml<T>::reshape(std::initializer_list<uli> new_shape) {
  reshape(array_mml<uli>(new_shape));
}

template <typename T> void Tensor_mml<T>::reverse_buffer() {
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
std::shared_ptr<Tensor<T>>
Tensor_mml<T>::slice(std::initializer_list<uli> slice_indices) {
  auto slice_indices_array = array_mml<uli>(slice_indices);
  return slice(slice_indices_array);
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor_mml<T>::slice(array_mml<uli> &slice_indices) {
  if (!valid_slice_indices(slice_indices)) {
    throw std::invalid_argument("Invalid slice indices");
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

  // Create a shallow std::copy of the data that the sliced tensor will use.
  std::shared_ptr<T[]> data_ptr(this->data.get(), [](T *) { /* noop */ });
  array_mml<T> data_shallow_copy = array_mml<T>(data_ptr, this->data.size());
  // Calculate the new indices offsets for the sliced tensor.
  array_mml<uli> new_slice_offsets =
      compute_slice_offsets(slice_indices, slice_shape);

  // Return a new Tensor but with a shallow std::copy of the data.
  std::shared_ptr<Tensor<T>> sliced_tensor = std::make_shared<Tensor_mml<T>>(
      slice_shape, data_shallow_copy, new_slice_offsets);

  return sliced_tensor;
}

template <typename T> bool Tensor_mml<T>::is_matrix() const {
  return this->shape.size() == 2;
}

template <typename T>
bool Tensor_mml<T>::matrix_match(const Tensor<T> &other) const {
  if (!this->is_matrix() || !other.is_matrix()) {
    return false;
  }
  return this->get_shape()[1] == other.get_shape()[0];
}

template <typename T>
bool Tensor_mml<T>::operator==(const Tensor<T> &other) const {
  if (this->get_size() != other.get_size())
    return false;
  if (this->get_shape() != other.get_shape())
    return false;

  for (uli i = 0; i < this->size; i++) {
    if ((*this)[i] != other[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Tensor_mml<T>::operator!=(const Tensor<T> &other) const {
  return !(*this == other);
}

template <typename T> const array_mml<uli> &Tensor_mml<T>::get_shape() const {
  return this->shape;
}

template <typename T> const array_mml<uli> &Tensor_mml<T>::get_offsets() const {
  return this->indices_offsets;
}

template <typename T> uli Tensor_mml<T>::get_size() const { return this->size; }

template <typename T>
const T &Tensor_mml<T>::operator[](array_mml<uli> &indices) const {
  if (!valid_indices(indices))
    throw std::invalid_argument("Invalid Tensor indices");
  if (this->slice_offsets.has_value())
    return this->data[index_to_slice_index(indices_to_1d_index(indices))];
  return this->data[indices_to_1d_index(indices)];
}

template <typename T> T &Tensor_mml<T>::operator[](array_mml<uli> &indices) {
  if (!valid_indices(indices))
    throw std::invalid_argument("Invalid Tensor indices");
  if (this->slice_offsets.has_value())
    return this->data[index_to_slice_index(indices_to_1d_index(indices))];
  return this->data[indices_to_1d_index(indices)];
}

template <typename T>
const T &Tensor_mml<T>::operator[](std::initializer_list<uli> indices) const {
  auto indices_array = array_mml<uli>(indices);
  return (*this)[indices_array];
}

template <typename T>
T &Tensor_mml<T>::operator[](std::initializer_list<uli> indices) {
  auto indices_array = array_mml<uli>(indices);
  return (*this)[indices_array];
}

template <typename T> const T &Tensor_mml<T>::operator[](uli index) const {
  if (!valid_index(index))
    throw std::invalid_argument("Invalid Tensor index");
  if (this->slice_offsets.has_value())
    return this->data[index_to_slice_index(index)];
  return this->data[index];
}

template <typename T> T &Tensor_mml<T>::operator[](uli index) {
  if (!valid_index(index))
    throw std::invalid_argument("Invalid Tensor index");
  if (this->slice_offsets.has_value())
    return this->data[index_to_slice_index(index)];
  return this->data[index];
}

template <typename T> void Tensor_mml<T>::fill(T value) {
  this->data.fill(value);
}

template <typename T>
std::shared_ptr<Tensor<T>>
Tensor_mml<T>::transpose(std::optional<uli> dim0,
                         std::optional<uli> dim1) const {
  uli rank = this->shape.size();
  uli d0 = dim0.value_or(rank > 1 ? rank - 2 : 0);
  uli d1 = dim1.value_or(rank > 1 ? rank - 1 : 0);

  if (d0 >= rank || d1 >= rank) {
    throw std::invalid_argument("Transpose dimensions out of range");
  }

  if (d0 == d1) {
    return this->copy();
  }

  array_mml<uli> new_shape = this->shape;
  std::swap(new_shape[d0], new_shape[d1]);

  auto transposed = std::make_shared<Tensor_mml<T>>(new_shape);

  if (rank == 2 && d0 == 0 && d1 == 1) {
    // Optimize the common 2D matrix transpose case
    for (uli i = 0; i < this->shape[0]; i++) {
      for (uli j = 0; j < this->shape[1]; j++) {
        (*transposed)[{j, i}] = (*this)[{i, j}];
      }
    }
  } else {
    std::function<void(array_mml<uli> &, uli)> transpose_recursive;
    transpose_recursive = [&](array_mml<uli> &indices, uli dim) {
      if (dim == rank) {
        array_mml<uli> transposed_indices = indices;
        std::swap(transposed_indices[d0], transposed_indices[d1]);
        (*transposed)[transposed_indices] = (*this)[indices];
        return;
      }

      for (uli i = 0; i < this->shape[dim]; ++i) {
        indices[dim] = i;
        transpose_recursive(indices, dim + 1);
      }
    };

    array_mml<uli> indices(rank);
    indices.fill(0);
    transpose_recursive(indices, 0);
  }

  return transposed;
};

template <typename T>
bool Tensor_mml<T>::is_broadcastable_to(
    const array_mml<uli> &target_shape) const {
  const array_mml<uli> &current_shape = this->shape;

  size_t i = current_shape.size();
  size_t j = target_shape.size();

  while (i > 0 && j > 0) {
    i--;
    j--;
    // Dimensions must either be std::equal or one must be 1
    if (current_shape[i] != target_shape[j] && current_shape[i] != 1 &&
        target_shape[j] != 1) {
      return false;
    }
  }

  // If the current tensor has remaining dimensions, they must all be 1
  while (i > 0) {
    i--;
    if (current_shape[i] != 1)
      return false;
  }

  return true;
};

template <typename T>
std::shared_ptr<Tensor<T>>
Tensor_mml<T>::broadcast_to(const array_mml<uli> &target_shape) const {
  if (this->shape == target_shape) {
    return this->copy();
  }

  if (!is_broadcastable_to(target_shape)) {
    throw std::invalid_argument("Cannot broadcast tensor to target shape");
  }

  auto result = std::make_shared<Tensor_mml<T>>(target_shape);
  const array_mml<uli> &current_shape = this->shape;
  uli rank_diff = target_shape.size() - current_shape.size();

  array_mml<uli> target_indices(target_shape.size());
  target_indices.fill(0);

  array_mml<uli> source_indices(current_shape.size());
  source_indices.fill(0);

  std::function<void(uli)> fill_broadcast = [&](uli dim) {
    if (dim == target_shape.size()) {
      for (uli i = 0; i < current_shape.size(); ++i) {
        uli target_i = i + rank_diff;
        source_indices[i] =
            (current_shape[i] == 1) ? 0 : target_indices[target_i];
      }

      (*result)[target_indices] = (*this)[source_indices];
      return;
    }

    for (uli i = 0; i < target_shape[dim]; ++i) {
      target_indices[dim] = i;
      fill_broadcast(dim + 1);
    }
  };

  fill_broadcast(0);
  return result;
};

template <typename T>
array_mml<uli> Tensor_mml<T>::compute_indices_offsets() const {
  const uli shape_size = this->shape.size();
  array_mml<uli> computed_offsets(shape_size);

  if (shape_size == 0) {
    // Scalar tensor: no offsets needed
    return computed_offsets;
  }

  computed_offsets[shape_size - 1] = 1;

  // Fill in offsets backwards
  for (int i = static_cast<int>(shape_size) - 2; i >= 0; --i) {
    computed_offsets[i] = this->shape[i + 1] * computed_offsets[i + 1];
  }

  return computed_offsets;
}

template <typename T>
array_mml<uli>
Tensor_mml<T>::compute_slice_offsets(array_mml<uli> &slice_indices,
                                     array_mml<uli> &slice_shape) const {
  uli jump_rows = 0;
  uli i_stride = 1;
  uli i_add = 0;

  // If the slice shape is odd, we need to adjust the i_stride and i_add to
  // account for columns
  if (slice_shape.size() % 2 != 0) {
    i_stride = this->shape[this->shape.size() - 1];
    i_add = slice_indices[slice_indices.size() - 1];
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

template <typename T> uli Tensor_mml<T>::compute_size() const {
  if (shape.size() == 0) {
    return 1; // Scalar tensor has 1 value
  }

  return std::accumulate(this->shape.begin(), this->shape.end(), 1,
                         std::multiplies<uli>());
}

template <typename T>
bool Tensor_mml<T>::valid_shape(const array_mml<uli> &new_shape) const {
  return std::accumulate(new_shape.begin(), new_shape.end(), 1,
                         std::multiplies<uli>()) == this->get_size();
}

template <typename T> bool Tensor_mml<T>::valid_index(uli index) const {
  return index < this->get_size();
}

template <typename T>
bool Tensor_mml<T>::valid_indices(const array_mml<uli> &indices) const {
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

template <typename T> uli Tensor_mml<T>::index_to_slice_index(uli index) const {
  if (!this->slice_offsets.has_value())
    throw std::invalid_argument("No slice offsets available");

  uli jump_rows = this->slice_offsets
                      .value()[0]; // How many rows should we jump in 1d buffer
  uli i_stride =
      this->slice_offsets.value()[1]; // How big is the stride per i index
  uli i_add = this->slice_offsets
                  .value()[2]; // How much should we add to i to offset it

  uli slice_index = jump_rows + index * i_stride + i_add;
  return slice_index;
}

template <typename T>
bool Tensor_mml<T>::valid_slice_indices(
    const array_mml<uli> &slice_indices) const {
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

// Convenience initializers

template <typename T>
std::shared_ptr<Tensor<T>>
tensor_mml_p(const std::initializer_list<uli> shape) {
  auto t = std::make_shared<Tensor_mml<T>>(shape);
  return t;
}

template <typename T>
std::shared_ptr<Tensor<T>> tensor_mml_p(const std::initializer_list<uli> shape,
                                        const std::initializer_list<T> data) {
  auto t = std::make_shared<Tensor_mml<T>>(shape, data);
  return t;
}