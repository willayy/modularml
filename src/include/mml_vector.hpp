#pragma once

#include "a_data_structure.hpp"
#include "array_mml.hpp"
#include "globals.hpp"

template <typename T>
class Vector_mml : public DataStructure<T> {
 public:
  // Override constructor
  explicit Vector_mml(int size) : DataStructure<T>() {
    this->data = vector<T>(size, 0);
  }

  // Override constructor
  explicit Vector_mml(const vector<T>& data) : DataStructure<T>() {
    this->data = data;
  }

  // Override move constructor
  Vector_mml(Vector_mml&& other) noexcept : DataStructure<T>(move(other)), data(move(other.data)) {}

  // Override copy constructor
  Vector_mml(const Vector_mml& other) : DataStructure<T>(other), data(vector<T>(other.data)) {}

  ~Vector_mml() override = default;

  void set_data(const array_mml<T>& new_data) override {
    vector<T> new_data_vector = vector<T>(new_data.size());
    for (int i = 0; i < new_data.size(); i++) {
      data[i] = new_data[i];
    }
    this->data = vector<T>(new_data_vector);
  }

  void set_zero() override {
    this->data = vector<T>(this->data.size(), 0);
  }

  array_mml<T> get_data() const override {
    return array_mml<T>(this->data);
  }

  int get_size() const override {
    return this->data.size();
  }

  const T& get_elem(int index) const override {
    if (!valid_index(index)) {
      throw out_of_range("Invalid Vector_mml index");
    } else {
      return this->data[index];
    }
  }

  T& get_mutable_elem(int index) override {
    if (!valid_index(index)) {
      throw out_of_range("Invalid Vector_mml index");
    } else {
      return this->data[index];
    }
  }

  shared_ptr<DataStructure<T>> clone() const override {
    return make_shared<Vector_mml<T>>(*this);
  }

 private:
  vector<T> data;

  /// @brief Check if the indices are valid. size of indices should be equal to the size of the shape. all elements of indices should be less than the corresponding element of the shape and greater than or equal to 0.
  /// @param indices The indices to check.
  /// @return True if the indices are valid, false otherwise.
  bool valid_index(const int index) const {
    if (const int size = this->data.size(); 0 > index || index >= size) {
      return false;
    }
    return true;
  }
};