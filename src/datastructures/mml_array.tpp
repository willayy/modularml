#pragma once

#include "datastructures/mml_array.hpp"

template <typename T>
array_mml<T>::array_mml(uli size)
    : d_size(size) {

    
#ifdef ALIGN_TENSORS
  size_t alignment = MEMORY_ALIGNMENT; // Gets set during compilation

  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) {
    throw std::bad_alloc();
  }

  data = std::shared_ptr<T[]>(static_cast<T*>(ptr), [](T* ptr) {
    free(ptr);
  });
#else
      data = std::shared_ptr<T[]>(new T[size]);
#endif
} 

template <typename T>
array_mml<T>::array_mml(initializer_list<T> data)
    : data(make_shared<T[]>(data.size())), d_size(data.size()) {
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(vector<T> &data)
    : data(make_shared<T[]>(data.size())), d_size(data.size()) {
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(const vector<T> &data)
    : data(make_shared<T[]>(data.size())), d_size(data.size()) {
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(shared_ptr<T[]> data, uli size)
    : data(data), d_size(size) {}

template <typename T>
array_mml<T>::array_mml(const array_mml &other)
    : data(make_shared<T[]>(other.d_size)), d_size(other.d_size) {
  copy(other.data.get(), other.data.get() + other.d_size, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(array_mml &&other) noexcept
    : data(move(other.data)), d_size(other.d_size) {
  other.d_size = 0;
}

template <typename T> uli array_mml<T>::size() const { return this->d_size; }

template <typename T> T &array_mml<T>::operator[](uli index) {
  if (index >= this->d_size) {
    throw out_of_range("Invalid array_mml index");
  } else {
    return this->data[index];
  }
}

template <typename T> const T &array_mml<T>::operator[](uli index) const {
  if (index >= this->d_size) {
    throw out_of_range("Invalid array_mml index");
  } else {
    return this->data[index];
  }
}

template <typename T>
array_mml<T> &array_mml<T>::operator=(const array_mml &other) {
  if (this != &other) {
    std::ranges::copy(other, this->data.get());
    this->d_size = other.d_size;
  }
  return *this;
}

template <typename T>
array_mml<T> array_mml<T>::subarray(uli start, uli end) const {
  if (start >= this->d_size || end > this->d_size || start > end) {
    throw out_of_range("Invalid array_mml subarray index");
  }
  array_mml new_array(end - start);
  copy(this->data.get() + start, this->data.get() + end, new_array.data.get());
  return new_array;
}

template <typename T>
bool array_mml<T>::operator==(const array_mml &other) const {
  if (this->d_size != other.d_size) {
    return false;
  }
  return equal(this->begin(), this->end(), other.begin());
}

template <typename T>
bool array_mml<T>::operator!=(const array_mml &other) const {
  return !(*this == other);
}

template <typename T> string array_mml<T>::to_string() const {
  string str = "[";
  // if longer than 50 print first 10 then ... then last 10
  if (this->size() > 50) {
    for (uli i = 0; i < 10; i++) {
      str += std::to_string(this->data[i]);
      str += ", ";
    }
    str += "..., ";
    for (uli i = this->size() - 10; i < this->size(); i++) {
      str += std::to_string(this->data[i]);
      if (i != this->size() - 1) {
        str += ", ";
      }
    }
  } else {
    for (uli i = 0; i < this->size(); i++) {
      str += std::to_string(this->data[i]);
      if (i != this->size() - 1) {
        str += ", ";
      }
    }
  }
  str += "]";
  return str;
}

template <typename T> T *array_mml<T>::begin() { return this->data.get(); }

template <typename T> const T *array_mml<T>::begin() const {
  return this->data.get();
}

template <typename T> T *array_mml<T>::end() {
  return this->data.get() + this->d_size;
}

template <typename T> const T *array_mml<T>::end() const {
  return this->data.get() + this->d_size;
}

template <typename T> T *array_mml<T>::get() { return this->data.get(); }

template <typename T> const T *array_mml<T>::get() const {
  return this->data.get();
}

template <typename T> void array_mml<T>::fill(const T &value) {
  std::ranges::fill(*this, value);
}