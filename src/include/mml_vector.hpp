#pragma once
#include <memory>
#include <numeric>
#include <stdexcept>

#include "a_data_structure.hpp"
#include "globals.hpp"

template <typename T>
class Vector_mml : public DataStructure<T> {
 public:
  Vector_mml(Vec<int> const& shape, Vec<T> data) : DataStructure<T>(), data(data), shape(shape) {
    this->offsets = compute_offsets();
  }

  explicit Vector_mml(Vec<int> const& shape) : DataStructure<T>(), shape(shape) {
    const int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    this->data = Vec<T>(size, 0);
    this->offsets = compute_offsets();
  }

  // Override move constructor
  Vector_mml(Vector_mml&& other) noexcept
      : data(std::move(other.data)), shape(std::move(other.shape)), offsets(std::move(other.offsets)) {}

  // Override copy constructor
  Vector_mml(const Vector_mml& other)
      : data(other.data), shape(other.shape), offsets(other.offsets) {}

  ~Vector_mml() override = default;

  void set_data(const Vec<T> new_data) override {
    this->data = new_data;
  }

  void set_zero() override {
    this->data = Vec<T>(this->data.size(), 0);
  }

  const Vec<int>& get_shape() const override {
    return this->shape;
  }

  const Vec<T>& get_raw_data() const override {
    return this->data;
  }

  int get_data_size() const override {
    return this->data.size();
  }

  String get_shape_str() const override {
    return "[" + std::to_string(this->data.size()) + "]";
  }

  const T& get_elem(Vec<int> indices) const override {
    if (!valid_index(indices)) {
      throw std::out_of_range("Invalid index");
    } else {
      return this->data[index_with_offset(indices)];
    }
  }

  T& get_mutable_elem(Vec<int> indices) override {
    if (!valid_index(indices)) {
      throw std::out_of_range("Invalid index");
    } else {
      return this->data[index_with_offset(indices)];
    }
  }

  std::unique_ptr<DataStructure<T>> clone() const override {
    return std::make_unique<Vector_mml<T>>(*this);
  }

 private:
  Vec<T> data;
  Vec<int> shape;
  Vec<int> offsets;

  /// @brief Check if the indices are valid. size of indices should be equal to the size of the shape. all elements of indices should be less than the corresponding element of the shape and greater than or equal to 0.
  /// @param indices The indices to check.
  /// @return True if the indices are valid, false otherwise.
  bool valid_index(const Vec<int>& indices) const {
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
  int index_with_offset(Vec<int> indices) const {
    auto index = 0;
    const auto size = static_cast<int>(shape.size());
    for (int i = 0; i < size; i++) {
      index += (indices[i]) * this->offsets[i];
    }
    return index;
  }

  /// @brief Row-major offsets for the data structure.
  /// @return a vector of integers representing the offsets.
  Vec<int> compute_offsets() const {
    const int size = static_cast<int>(shape.size());
    auto computed_offsets = Vec<int>(size, 1);
    for (int i = size - 2; i >= 0; i--) {
      computed_offsets[i] = computed_offsets[i + 1] * this->shape[i];
    }
    return computed_offsets;
  }
};