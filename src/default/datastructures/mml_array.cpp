#include "datastructures/mml_array.hpp"

template <typename T>
array_mml<T>::array_mml(size_t size) : d_size(size) {
#ifdef ALIGN_TENSORS
  this->data = alloc_aligned_memory<T>(d_size);
#else
  this->data = std::shared_ptr<T[]>(new T[size]);
#endif
}

template <typename T>
array_mml<T>::array_mml(std::initializer_list<T> data) : d_size(data.size()) {
#ifdef ALIGN_TENSORS
  this->data = alloc_aligned_memory<T>(d_size);
#else
  this->data = std::shared_ptr<T[]>(new T[d_size]);
#endif

  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(std::vector<T> &data) : d_size(data.size()) {
#ifdef ALIGN_TENSORS
  this->data = alloc_aligned_memory<T>(d_size);
#else
  this->data = std::shared_ptr<T[]>(new T[d_size]);
#endif
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(const std::vector<T> &data) : d_size(data.size()) {
#ifdef ALIGN_TENSORS
  this->data = alloc_aligned_memory<T>(d_size);
#else
  this->data = std::shared_ptr<T[]>(new T[d_size]);
#endif
  std::ranges::copy(data, this->data.get());
}

template <typename T>
array_mml<T>::array_mml(std::shared_ptr<T[]> data, size_t size)
    : data(data), d_size(size) {
#ifdef ALIGN_TENSORS
  // Alert if the passed memory is not aligned
  if (reinterpret_cast<uintptr_t>(data.get()) % MEMORY_ALIGNMENT != 0) {
    throw std::runtime_error("Data is not aligned correctly.");
  }
#endif
}

template <typename T>
array_mml<T>::array_mml(const array_mml &other) : d_size(other.d_size) {
#ifdef ALIGN_TENSORS
  this->data = alloc_aligned_memory<T>(d_size);
#else
  this->data = std::make_shared<T[]>(d_size);
#endif
  std::copy(other.data.get(), other.data.get() + other.d_size,
            this->data.get());
}

template <typename T>
array_mml<T>::array_mml(array_mml &&other) noexcept
    : data(std::move(other.data)), d_size(other.d_size) {
  other.d_size = 0;
}

template <typename T>
size_t array_mml<T>::size() const {
  return this->d_size;
}

template <typename T>
T &array_mml<T>::operator[](size_t index) {
  if (index >= this->d_size) {
    throw std::out_of_range(
        "Invalid array_mml index: " + std::to_string(index) +
        ". Array size: " + std::to_string(this->d_size));
  } else {
    return this->data[index];
  }
}

template <typename T>
const T &array_mml<T>::operator[](size_t index) const {
  if (index >= this->d_size) {
    throw std::out_of_range(
        "Invalid array_mml index: " + std::to_string(index) +
        ". Array size: " + std::to_string(this->d_size));
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
array_mml<T> array_mml<T>::subarray(size_t start, size_t end) const {
  if (start >= this->d_size || end > this->d_size || start > end) {
    throw std::out_of_range("Invalid array_mml subarray index");
  }
  array_mml new_array(end - start);
  std::copy(this->data.get() + start, this->data.get() + end,
            new_array.data.get());
  return new_array;
}

template <typename T>
bool array_mml<T>::operator==(const array_mml &other) const {
  if (this->d_size != other.d_size) {
    return false;
  }
  return std::equal(this->begin(), this->end(), other.begin());
}

template <typename T>
std::string array_mml<T>::to_string() const {
  std::string str = "[";
  // if longer than 50 print first 10 then ... then last 10
  if (this->size() > 50) {
    for (size_t i = 0; i < 10; i++) {
      str += std::to_string(this->data[i]);
      str += ", ";
    }
    str += "..., ";
    for (size_t i = this->size() - 10; i < this->size(); i++) {
      str += std::to_string(this->data[i]);
      if (i != this->size() - 1) {
        str += ", ";
      }
    }
  } else {
    for (size_t i = 0; i < this->size(); i++) {
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
T *array_mml<T>::begin() {
  return this->data.get();
}

template <typename T>
const T *array_mml<T>::begin() const {
  return this->data.get();
}

template <typename T>
T *array_mml<T>::end() {
  return this->data.get() + this->d_size;
}

template <typename T>
const T *array_mml<T>::end() const {
  return this->data.get() + this->d_size;
}

template <typename T>
T *array_mml<T>::get() {
  return this->data.get();
}

template <typename T>
const T *array_mml<T>::get() const {
  return this->data.get();
}

template <typename T>
void array_mml<T>::fill(const T &value) {
  std::ranges::fill(*this, value);
}

#define TYPE(DT) _ARRAY_MML(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE
