#pragma once

#include "globals.hpp"

/// @brief Array class mimicking the std::array class but without the size being a template parameter.
/// @tparam T the type of the array.
template <typename T>
class array_ml {
    public:
    /// @brief Constructor for array_ml class.
    /// @param size The size of the array.
    explicit array_ml(int size) : data(make_unique<T[]>(size)), size(size) {}
    
    /// @brief Copy constructor for array_ml class.
    array_ml(const array_ml& other) = default;

    /// @brief Move constructor for array_ml class.
    array_ml(array_ml&& other) noexcept = default;

    /// @brief Destructor for array_ml class.
    ~array_ml() = default;

    /// @brief Get the size of the array, the number of elements in the array.
    /// @return The size of the array.
    int size() const {
        return this->size;
    }

    /// @brief Get an element from the array using a single-dimensional index.
    /// @param index The index of the element to get.
    /// @return The element at the given index.
    T& operator[](int index) {
        if (index < 0 || index >= this->size) {
            throw out_of_range("Invalid array_ml index");
        } else {
            return this->data[index];
        }
    }

    /// @brief Get an element from the array using a single-dimensional index.
    /// @param index The index of the element to get.
    /// @return The element at the given index.
    const T& operator[](int index) const {
        if (index < 0 || index >= this->size) {
            throw out_of_range("Invalid array_ml index");
        } else {
            return this->data[index];
        }
    }

    private:
        unique_ptr<T[]> data;
        int size;
};