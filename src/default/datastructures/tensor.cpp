#include "datastructures/tensor.hpp"

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<size_t> shape,
                  const size_t jump_indexes, const size_t jump_columns,
                  const size_t jump_rows, const bool sliced)
    : shape(shape),
      jump_indexes(jump_indexes),
      jump_columns(jump_columns),
      jump_rows(jump_rows),
      sliced(sliced) {
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->size);
  this->data.fill(T(0));
}

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<size_t> shape,
                  const std::initializer_list<T> data,
                  const size_t jump_indexes, const size_t jump_columns,
                  const size_t jump_rows, const bool sliced)
    : shape(shape),
      data(data),
      jump_indexes(jump_indexes),
      jump_columns(jump_columns),
      jump_rows(jump_rows),
      sliced(sliced) {
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
}

template <typename T>
Tensor<T>::Tensor(const array_mml<size_t> &shape, const size_t jump_indexes,
                  const size_t jump_columns, const size_t jump_rows,
                  const bool sliced)
    : shape(shape),
      jump_indexes(jump_indexes),
      jump_columns(jump_columns),
      jump_rows(jump_rows),
      sliced(sliced) {
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
  this->data = array_mml<T>(this->size);
  this->data.fill(T(0));
}

template <typename T>
Tensor<T>::Tensor(const array_mml<size_t> &shape, const array_mml<T> &data,
                  const size_t jump_indexes, const size_t jump_columns,
                  const size_t jump_rows, const bool sliced)
    : shape(shape),
      data(data),
      jump_indexes(jump_indexes),
      jump_columns(jump_columns),
      jump_rows(jump_rows),
      sliced(sliced) {
  this->indices_offsets = compute_indices_offsets();
  this->size = compute_size();
}

template <typename T>
Tensor<T>::Tensor(Tensor &&other) noexcept {
  this->shape = std::move(other.shape);
  this->indices_offsets = std::move(other.indices_offsets);
  this->size = other.size;
  this->jump_indexes = other.jump_indexes;
  this->jump_columns = other.jump_columns;
  this->jump_rows = other.jump_rows;
  this->sliced = other.sliced;
  this->data = std::move(other.data);
}

template <typename T>
Tensor<T>::Tensor(const Tensor &other) {
  this->shape = array_mml<size_t>(other.shape);
  this->indices_offsets = array_mml<size_t>(other.indices_offsets);
  this->data = array_mml<T>(other.data);
  this->size = other.size;
  this->jump_indexes = other.jump_indexes;
  this->jump_columns = other.jump_columns;
  this->jump_rows = other.jump_rows;
  this->sliced = other.sliced;
}

template <typename T>
const array_mml<T> &Tensor<T>::get_data() const {
  return this->data;
}

template <typename T>
array_mml<T> &Tensor<T>::get_raw_data() {
  return this->data;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
  if (this != &other) {
    auto other_cast = dynamic_cast<const Tensor<T> &>(other);
    this->shape = array_mml<size_t>(other_cast.shape);
    this->indices_offsets = array_mml<size_t>(other_cast.indices_offsets);
    this->data = array_mml<T>(other_cast.data);
    this->size = other_cast.size;
    this->jump_indexes = other_cast.jump_indexes;
    this->jump_columns = other_cast.jump_columns;
    this->jump_rows = other_cast.jump_rows;
    this->sliced = other_cast.sliced;
  }
  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other) noexcept {
  if (this != &other) {
    auto other_cast = dynamic_cast<Tensor<T> &&>(std::move(other));
    this->data = std::move(other_cast.data);
    this->shape = std::move(other_cast.shape);
    this->indices_offsets = std::move(other_cast.indices_offsets);
    this->size = other_cast.size;
    this->jump_indexes = other_cast.jump_indexes;
    this->jump_columns = other_cast.jump_columns;
    this->jump_rows = other_cast.jump_rows;
    this->sliced = other_cast.sliced;
  }
  return *this;
}

template <typename T>
std::string Tensor<T>::to_string() const {
  std::string base = std::string("Tensor<") + typeid(T).name() + "> ";
  std::string ptr_str =
      "Pointer: " + std::to_string(reinterpret_cast<uintptr_t>(this));
  std::string shape_str = "Shape: " + this->shape.to_string();
  std::string size_str = "Size: " + std::to_string(this->size);
  std::string data_str = "Data: ";
  if (this->size > 30) {
    std::string first_10 = "[";
    std::string last_10 = "";
    for (size_t i = 0; i < 9; i++) {
      first_10 += std::to_string((*this)[i]) + ", ";
      last_10 += std::to_string((*this)[this->size - i - 1]) + ", ";
    }
    first_10 += std::to_string((*this)[9]) + ", ... ";
    last_10 += std::to_string((*this)[this->size - 10]) + "]";

    data_str += first_10 + " ... " + last_10;
  } else {
    data_str += "[";
    for (size_t i = 0; i < this->size - 1; i++) {
      data_str += std::to_string((*this)[i]) + ", ";
    }
    data_str += std::to_string((*this)[this->size - 1]) + "]";
  }
  return base + ptr_str + ", " + shape_str + ", " + size_str + ", " + data_str;
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::copy() const {
  return std::make_shared<Tensor<T>>(*this);
}

template <typename T>
void Tensor<T>::reshape(const array_mml<size_t> &new_shape) {
  if (!valid_shape(new_shape)) throw std::invalid_argument("Invalid shape");
  this->shape = array_mml<size_t>(new_shape);
  this->indices_offsets = compute_indices_offsets();
}

template <typename T>
void Tensor<T>::reshape(std::initializer_list<size_t> new_shape) {
  reshape(array_mml<size_t>(new_shape));
}

template <typename T>
void Tensor<T>::reverse_buffer() {
  size_t i = 0;
  size_t j = this->size - 1;
  while (i < j) {
    T temp = this->data[i];
    this->data[i] = this->data[j];
    this->data[j] = temp;
    i++;
    j--;
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::slice(
    std::initializer_list<size_t> slice_indices) {
  auto slice_indices_array = array_mml<size_t>(slice_indices);
  return slice(slice_indices_array);
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::slice(array_mml<size_t> &slice_indices) {
  if (!valid_slice_indices(slice_indices))
    throw std::invalid_argument("Invalid slice indices");

  // Find the value of jump indexes
  size_t slice_jump_indexes = this->sliced ? this->jump_indexes : 0;

  // Calculate the jump indexes, ignore tensors that are matrices
  if (this->shape.size() > 2) {
    size_t i = 0;
    do {
      slice_jump_indexes += this->indices_offsets[i] * slice_indices[i];
    } while (i++, i < slice_indices.size() - 1);
  }

  // Create the buffer
  std::shared_ptr<T[]> data_ptr(this->data.get(), [](T *) { /* NOOP */ });
  auto shared_buffer = array_mml<T>(data_ptr, this->data.size());

  // New shape and jump row/col for column slices.
  array_mml<size_t> slice_shape(this->shape.size() - slice_indices.size());
  size_t slice_jump_columns = 0;
  size_t slice_jump_rows = 1;
  // Is the slice a column slice?
  if (slice_indices.size() == this->shape.size() - 1) {
    slice_jump_columns = slice_indices[slice_indices.size() - 1];
    slice_jump_rows = this->shape[this->shape.size() - 1];
    slice_shape[0] = this->shape[this->shape.size() - 2];
  } else {
    // Create the slice shape
    for (size_t i = 0; i < slice_shape.size(); i++) {
      slice_shape[i] = this->shape[i + slice_indices.size()];
    }
  }

  auto sliced_tensor = std::make_shared<Tensor<T>>(
      slice_shape, shared_buffer, slice_jump_indexes, slice_jump_columns,
      slice_jump_rows, true);

  return sliced_tensor;
}

template <typename T>
bool Tensor<T>::is_matrix() const {
  return this->shape.size() == 2;
}

template <typename T>
bool Tensor<T>::operator==(const Tensor<T> &other) const {
  if (this->get_size() != other.get_size()) return false;
  if (this->get_shape() != other.get_shape()) return false;

  for (size_t i = 0; i < this->size; i++) {
    if ((*this)[i] != other[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
const array_mml<size_t> &Tensor<T>::get_shape() const {
  return this->shape;
}

template <typename T>
const array_mml<size_t> &Tensor<T>::get_offsets() const {
  return this->indices_offsets;
}

template <typename T>
size_t Tensor<T>::get_size() const {
  return this->size;
}

template <typename T>
const T &Tensor<T>::operator[](array_mml<size_t> &indices) const {
  if (!valid_indices(indices))
    throw std::invalid_argument("Invalid Tensor indices");
  if (this->sliced)
    return this->data[index_to_offset_1d_index(indices_to_1d_index(indices))];
  return this->data[indices_to_1d_index(indices)];
}

template <typename T>
T &Tensor<T>::operator[](array_mml<size_t> &indices) {
  if (!valid_indices(indices))
    throw std::invalid_argument("Invalid Tensor indices");
  if (this->sliced)
    return this->data[index_to_offset_1d_index(indices_to_1d_index(indices))];
  return this->data[indices_to_1d_index(indices)];
}

template <typename T>
const T &Tensor<T>::operator[](std::initializer_list<size_t> indices) const {
  auto indices_array = array_mml<size_t>(indices);
  return (*this)[indices_array];
}

template <typename T>
T &Tensor<T>::operator[](std::initializer_list<size_t> indices) {
  auto indices_array = array_mml<size_t>(indices);
  return (*this)[indices_array];
}

template <typename T>
const T &Tensor<T>::operator[](size_t index) const {
  if (!valid_index(index)) throw std::invalid_argument("Invalid Tensor index");
  if (this->sliced) return this->data[index_to_offset_1d_index(index)];
  return this->data[index];
}

template <typename T>
T &Tensor<T>::operator[](size_t index) {
  if (!valid_index(index)) throw std::invalid_argument("Invalid Tensor index");
  if (this->sliced) return this->data[index_to_offset_1d_index(index)];
  return this->data[index];
}

template <typename T>
void Tensor<T>::fill(T value) {
  this->data.fill(value);
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::transpose(
    std::optional<size_t> dim0, std::optional<size_t> dim1) const {
  size_t rank = this->shape.size();
  size_t d0 = dim0.value_or(rank > 1 ? rank - 2 : 0);
  size_t d1 = dim1.value_or(rank > 1 ? rank - 1 : 0);

  if (d0 >= rank || d1 >= rank) {
    throw std::invalid_argument("Transpose dimensions out of range");
  }

  if (d0 == d1) {
    return this->copy();
  }

  array_mml<size_t> new_shape = this->shape;
  std::swap(new_shape[d0], new_shape[d1]);

  auto transposed = std::make_shared<Tensor<T>>(new_shape);

  if (rank == 2 && d0 == 0 && d1 == 1) {
    // Optimize the common 2D matrix transpose case using blocking
    const size_t block_size = 128;
    size_t i, j, bi, bj;
    auto rows = this->shape[0];
    auto cols = this->shape[1];
    for (i = 0; i < rows; i += block_size) {
      for (j = 0; j < cols; j += block_size) {
        for (bi = i; bi < std::min(i + block_size, rows); bi++) {
          for (bj = j; bj < std::min(j + block_size, cols); bj++) {
            (*transposed)[bj * rows + bi] = (*this)[bi * cols + bj];
          }
        }
      }
    }
  } else {
    std::function<void(array_mml<size_t> &, size_t)> transpose_recursive;
    transpose_recursive = [&](array_mml<size_t> &indices, size_t dim) {
      if (dim == rank) {
        array_mml<size_t> transposed_indices = indices;
        std::swap(transposed_indices[d0], transposed_indices[d1]);
        (*transposed)[transposed_indices] = (*this)[indices];
        return;
      }

      for (size_t i = 0; i < this->shape[dim]; ++i) {
        indices[dim] = i;
        transpose_recursive(indices, dim + 1);
      }
    };

    array_mml<size_t> indices(rank);
    indices.fill(0);
    transpose_recursive(indices, 0);
  }

  return transposed;
};

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::transpose(
    const std::vector<int> &perm) const {
  if (perm.size() != this->shape.size()) {
    throw std::invalid_argument(
        "Transpose: perm size must be equal to tensor rank");
  }

  // Checks that the perm vector is valid
  std::vector<bool> seen(perm.size(), false);
  for (size_t p : perm) {
    if (p >= perm.size() || seen[p]) {
      throw std::invalid_argument(
          "Transpose: invalid or duplicate entry in perm");
    }
    seen[p] = true;
  }

  array_mml<size_t> new_shape(perm.size());

  for (size_t i = 0; i < perm.size(); ++i) {
    size_t val = shape[perm[i]];
    new_shape[i] = val;
  }

  auto transposed = std::make_shared<Tensor<T>>(new_shape);
  // This recursive function remaps each element in the old tensor to the new
  // permutation based in the perm vector
  std::function<void(array_mml<size_t> &, size_t)> recurse;
  recurse = [&](array_mml<size_t> &indices, size_t dim) {
    if (dim == shape.size()) {
      array_mml<size_t> transposed_indices(perm.size());

      for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= indices.size()) {
          std::cerr << "ERROR: perm[" << i << "] = " << perm[i]
                    << " out of bounds for indices of size " << indices.size()
                    << std::endl;
          throw std::out_of_range("perm index out of bounds");
        }
        transposed_indices[i] = indices[perm[i]];
      }

      try {
        auto val = (*this)[indices];
        (*transposed)[transposed_indices] = val;
      } catch (const std::exception &e) {
        std::cerr << "Exception accessing tensor at ";
        for (size_t idx : indices) std::cerr << idx << " ";
        std::cerr << " or transposed at ";
        for (size_t idx : transposed_indices) std::cerr << idx << " ";
        std::cerr << ": " << e.what() << std::endl;
        throw;
      }

      return;
    }

    for (size_t i = 0; i < shape[dim]; ++i) {
      indices[dim] = i;
      recurse(indices, dim + 1);
    }
  };

  array_mml<size_t> indices(shape.size());
  indices.fill(0);
  recurse(indices, 0);

  return transposed;
}

template <typename T>
bool Tensor<T>::valid_broadcast_reshape_size(
    const array_mml<size_t> &target_shape) const {
  const array_mml<size_t> &current_shape = this->shape;

  size_t i = current_shape.size();
  size_t j = target_shape.size();

  while (i > 0 && j > 0) {
    i--;
    j--;
    // Dimensions must either be equal or one must be 1
    if (current_shape[i] != target_shape[j] && current_shape[i] != 1 &&
        target_shape[j] != 1) {
      return false;
    }
  }

  // If the current tensor has remaining dimensions, they must all be 1
  while (i > 0) {
    i--;
    if (current_shape[i] != 1) return false;
  }

  return true;
};

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::broadcast_reshape(
    const array_mml<size_t> &target_shape) const {
  if (this->shape == target_shape) return this->copy();

  if (!valid_broadcast_reshape_size(target_shape))
    throw std::invalid_argument("Cannot broadcast tensor to target shape");

  if (this->sliced) throw std::logic_error("Cannot broadcast a sliced tensor");

  // Caclulate how many times we should repeat the tensor
  size_t tensor_size = this->data.size();
  size_t target_size = std::accumulate(target_shape.begin(), target_shape.end(),
                                       1, std::multiplies<size_t>());
  size_t repeat_count = target_size / tensor_size;

  // Create the new buffer
  auto broadcasted_buffer = array_mml<T>(this->data.size() * repeat_count);
  for (size_t i = 0; i < repeat_count; i++) {
    for (size_t j = 0; j < tensor_size; j++) {
      broadcasted_buffer[i * tensor_size + j] = this->data[j];
    }
  }

  // Create the new tensor
  auto broadcasted_tensor =
      std::make_shared<Tensor<T>>(target_shape, broadcasted_buffer);
  return broadcasted_tensor;
};

template <typename T>
array_mml<size_t> Tensor<T>::compute_indices_offsets() const {
  const size_t shape_size = this->shape.size();
  array_mml<size_t> computed_offsets(shape_size);

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
size_t Tensor<T>::compute_size() const {
  if (shape.size() == 0) {
    return 1;  // Scalar tensor has 1 value
  }

  return std::accumulate(this->shape.begin(), this->shape.end(), 1,
                         std::multiplies<size_t>());
}

template <typename T>
bool Tensor<T>::valid_shape(const array_mml<size_t> &new_shape) const {
  return std::accumulate(new_shape.begin(), new_shape.end(), 1,
                         std::multiplies<size_t>()) == this->get_size();
}

template <typename T>
bool Tensor<T>::valid_index(size_t index) const {
  return index < this->get_size();
}

template <typename T>
bool Tensor<T>::valid_indices(const array_mml<size_t> &indices) const {
  if (indices.size() != this->shape.size()) {
    return false;
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= this->shape[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
size_t Tensor<T>::indices_to_1d_index(array_mml<size_t> indices) const {
  auto index = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    index += (indices[i]) * this->indices_offsets[i];
  }
  return index;
}

template <typename T>
size_t Tensor<T>::index_to_offset_1d_index(size_t index) const {
  if (!this->sliced) throw std::logic_error("Not a sliced tensor");
  return this->jump_indexes + index * this->jump_rows + this->jump_columns;
}

template <typename T>
bool Tensor<T>::valid_slice_indices(
    const array_mml<size_t> &slice_indices) const {
  if (slice_indices.size() >= this->shape.size()) {
    return false;
  }
  size_t slice_shape_dif = this->shape.size() - slice_indices.size();
  for (size_t i = 0; i < slice_indices.size(); i++) {
    if (slice_indices[i] >= this->shape[i + slice_shape_dif]) {
      return false;
    }
  }
  return true;
}

#define TYPE(DT) _TENSOR(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE
