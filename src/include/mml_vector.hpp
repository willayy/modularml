#pragma once
#include <numeric>
#include <stdexcept>

#include "a_data_structure.hpp"
#include "globals.hpp"

template <typename T>
class Vector_mml : public DataStructure<T> {
 public:
  Vector_mml(vec<int> const& shape, vec<T> data) : DataStructure<T>(), shape(shape) {
    this->data = data;
    this->offsets = compute_offsets();
  }

  explicit Vector_mml(vec<int> const& shape) : DataStructure<T>(), shape(shape) {
    const int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    this->data = vec<T>(size, 0);
    this->offsets = compute_offsets();
  }

  // Override move constructor
  Vector_mml(Vector_mml &&other) noexcept {
    this->data = other.data;
    this->shape = other.shape;
    this->offsets = other.offsets;
  }

  // Override copy constructor
  Vector_mml(const Vector_mml &other) {
    this->data = other.data;
    this->shape = other.shape;
    this->offsets = other.offsets;
  }

  ~Vector_mml() override = default;

  void set_data(const vec<T> new_data) override {
    this->data = new_data;
  }

  void set_zero() override {
    this->data = vec<T>(this->data.size(), 0);
  }

  const vec<int>& get_shape() const override {
    return this->shape;
  }

  const vec<T>& get_raw_data() const override {
    return this->data;
  }

  int get_data_size() const override {
    return this->data.size();
  }

  string get_shape_str() const override {
    return "[" + std::to_string(this->data.size()) + "]";
  }

  const T& get_elem(vec<int> indices) const override {
    if (!valid_index(indices)) {
      throw std::out_of_range("Invalid index");
    } else {
      return this->data[index_with_offset(indices)];
    }
  }

  T& get_mutable_elem(vec<int> indices) override {
    if (!valid_index(indices)) {
      throw std::out_of_range("Invalid index");
    } else {
      return this->data[index_with_offset(indices)];
    }
  }

 private:
  vec<T> data;
  vec<int> shape;
  vec<int> offsets;

  /// @brief Check if the indices are valid. size of indices should be equal to the size of the shape. all elements of indices should be less than the corresponding element of the shape and greater than or equal to 0.
  /// @param indices The indices to check.
  /// @return True if the indices are valid, false otherwise.
  bool valid_index(const vec<int>& indices) const {
    if (indices.size() != this->shape.size()) {
      return false;
    }
    const int size = static_cast<int>(shape.size());
    for (int i = 0; i < size; i++) {
      if (!(indices[i] < this->shape[i] && indices[i] >= 0)) {
        return false;
      }
    }
    return true;
  }

  /// @brief Calculates the index of an element in the flat vector containing the data.
  /// @param indices The indices to get the index for.
  /// @return The index.
  int index_with_offset(vec<int> indices) const {
    auto index = 0;
    const auto size = static_cast<int>(shape.size());
    for (int i = 0; i < size; i++) {
      index += (indices[i]) * this->offsets[i];
    }
    return index;
  }

  /// @brief Row-major offsets for the data structure.
  /// @return a vector of integers representing the offsets.
  vec<int> compute_offsets() const {
    const int size = static_cast<int>(shape.size());
    auto computed_offsets = vec<int>(size, 1);
    for (int i = size - 2; i >= 0; i--) {
      computed_offsets[i] = computed_offsets[i + 1] * this->shape[i];
    }
    return computed_offsets;
  }
};