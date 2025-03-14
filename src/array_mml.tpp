#pragma once

#include "array_mml.hpp"

template <typename T>
array_mml<T>::array_mml(unsigned long int size)
    : data(make_shared<T[]>(size)),
      d_size(size) {}

template <typename T>
array_mml<T>::array_mml(initializer_list<T> data)
    : data(make_shared<T[]>(data.size())),
      d_size(data.size()) {
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(vector<T>& data)
    : data(make_shared<T[]>(data.size())),
      d_size(data.size()) {
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(const vector<T>& data)
    : data(make_shared<T[]>(data.size())),
      d_size(data.size()) {
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(shared_ptr<T[]> data, unsigned long int size)
    : data(data),
      d_size(size) {
}

template <typename T>
array_mml<T>::array_mml(const array_mml& other)
    : data(make_shared<T[]>(other.d_size)),
      d_size(other.d_size) {
  copy(other.data.get(), other.data.get() + other.d_size, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(array_mml&& other) noexcept
    : data(move(other.data)),
      d_size(other.d_size) {
  other.d_size = 0;
}

template <typename T>
unsigned long int array_mml<T>::size() const {
  return this->d_size;
}

template <typename T>
T& array_mml<T>::operator[](unsigned long int index) {
  if (index >= this->d_size) {
    throw out_of_range("Invalid array_mml index");
  } else {
    return this->data[index];
  }
}

template <typename T>
const T& array_mml<T>::operator[](unsigned long int index) const {
  if (index >= this->d_size) {
    throw out_of_range("Invalid array_mml index");
  } else {
    return this->data[index];
  }
}

template <typename T>
array_mml<T>& array_mml<T>::operator=(const array_mml& other) {
  if (this != &other) {
    std::ranges::copy(other, this->data.get());
    this->d_size = other.d_size;
  }
  return *this;
}

template <typename T>
array_mml<T> array_mml<T>::subarray(unsigned long int start, unsigned long int end) const {
  if (start >= this->d_size || end > this->d_size || start > end) {
    throw out_of_range("Invalid array_mml subarray index");
  }
  array_mml new_array(end - start);
  copy(this->data.get() + start, this->data.get() + end, new_array.data.get());
  return new_array;
}

template <typename T>
array_mml<T> array_mml<T>::m_subarray(unsigned long int start, unsigned long int end) {
  if (start >= this->d_size || end > this->d_size || start > end) {
    throw out_of_range("Invalid array_mml subarray index");
  }
  auto data_ptr_address = this->data.get();
  array_mml new_array(this->data + start, this->data + end);
  return new_array;
}

template <typename T>
bool array_mml<T>::operator==(const array_mml& other) const {
  if (this->d_size != other.d_size) {
    return false;
  }
  return equal(this->begin(), this->end(), other.begin());
}

template <typename T>
bool array_mml<T>::operator!=(const array_mml& other) const {
  return !(*this == other);
}

template <typename T>
string array_mml<T>::to_string() const {
  string str = "[";
  // if longer than 50 print first 10 then ... then last 10
  if (this->size() > 50) {
    for (unsigned long int i = 0; i < 10; i++) {
      str += std::to_string(this->data[i]);
      str += ", ";
    }
    str += "..., ";
    for (unsigned long int i = this->size() - 10; i < this->size(); i++) {
      str += std::to_string(this->data[i]);
      if (i != this->size() - 1) {
        str += ", ";
      }
    }
  } else {
    for (unsigned long int i = 0; i < this->size(); i++) {
      str += std::to_string(this->data[i]);
      if (i != this->size() - 1) {
        str += ", ";
      }
    }
  }
  str += "]";
  return str;
}

template <typename T>
T* array_mml<T>::begin() {
  return this->data.get();
}

template <typename T>
const T* array_mml<T>::begin() const {
  return this->data.get();
}

template <typename T>
T* array_mml<T>::end() {
  return this->data.get() + this->d_size;
}

template <typename T>
const T* array_mml<T>::end() const {
  return this->data.get() + this->d_size;
}

template <typename T>
T* array_mml<T>::get() {
  return this->data.get();
}

template <typename T>
const T* array_mml<T>::get() const {
  return this->data.get();
}

template <typename T>
void array_mml<T>::fill(const T& value) {
  std::ranges::fill(*this, value);
}