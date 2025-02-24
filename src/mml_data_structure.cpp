#include "mml_data_structure.hpp"

template <typename T>
MML_DataStructure<T>::MML_DataStructure(const vec<int>& shape) {
    if (shape.size() <= 0) throw std::invalid_argument("Shape is invalid");

    this->shape = shape;

    // Calculate the size using the diemensions provided
    int data_size = 1;
    for (int i : shape) data_size *= i;
    this->data = vec<T>(data_size);
}

template <typename T>
MML_DataStructure<T>::MML_DataStructure(const vec<int>& shape, const vec<T> data) {
    if (shape.size() <= 0) throw std::invalid_argument("Shape isn't valid");

    int data_size = 1;
    for (int i : shape) data_size *= i;

    if (data_size != data.size()) throw std::invalid_argument("The size of the data and the shape does not match");

    this->shape = shape;
    this->data = data;
}

template <typename T>
T MML_DataStructure<T>::get(const vec<int>& indices) const {
    if (indices.size() == 0) throw std::invalid_argument("Invalid indices");
    if (indices.size() != this->shape.size()) throw std::invalid_argument("The diemensions of the indices does not match with the data structure");

    return this->data.at(calc_offset(indices));
}

template <typename T>
void MML_DataStructure<T>::set(const vec<int>& indices, T value) {
    if (indices.size() == 0) throw std::invalid_argument("Invalid indices");
    if (indices.size() != this->shape.size()) throw std::invalid_argument("The diemensions of the indices does not match with the data structure");

    this->data[calc_offset(indices)] = value;
}

template <typename T>
void MML_DataStructure<T>::set(const int index, T value) {
    if (indices.size() == 0) throw std::invalid_argument("Invalid indices");
    if (indices.size() != this->shape.size()) throw std::invalid_argument("The diemensions of the indices does not match with the data structure");

    this->data[index] = value;
}

template <typename T>
void MML_DataStructure<T>::set_data(const vec<T> new_data) {
    // TODO: Parallelize
    if (this->get_size() != data.size())
        throw std::invalid_argument("The size of the data provided mismatch with the size of the data structure");

    for (int i : this->data.size()) {
        this->data[i] = new_data[i];
    }
}

template <typename T>
void MML_DataStructure<T>::set_zero() {
    // TODO: Parallelize
    for (int i : this->data.size()) {
        this->data[i] = 0;
    }
}

template <typename T>
const vec<int> MML_DataStructure<T>::get_shape() const {
    return this->shape;
}

template <typename T>
const string MML_DataStructure<T>::get_shape_str() const {
    string result = "[";
    for (int i = 0; i < this->shape.size(); ++i) {
        if (i + 1 == this->shape.size())
            result += std::to_string(i);
        else
            result += (std::to_string(i) + ", ");
    }
    result += "]";
    return result;
}

template <typename T>
int MML_DataStructure<T>::get_size() const {
    return this->data.size();  // This assumes that the structure is populated by data
}

template <typename T>
bool MML_DataStructure<T>::equals(const MML_DataStructure<T>& other) const {
    if (this->data.size() == other->data.size()) {
        for (int i = 0; i < this->data.size(); ++i) {
            if (this->data.at(i) != other->data.size())  // Check the diemensions of the structures
                return false;
        }
        return true;
    }
    return false;
}

template <typename T>
int MML_DataStructure<T>::calc_offset(const vec<int>& indices) const {
    int offset = 0;
    int factor = 1;
    for (int i = indices.size() - 1; i >= 0; --i) {
        if (indices.at(i) < 0 || indices.at(i) >= this->shape.at(i)) { // Check that the index isn't out of range
            throw std::out_of_range("Index in indices vector are out of range " + std::to_string(i));
        }
        offset += indices.at(i) * factor;
        factor *= this->shape(i);
    }
    return offset;
}
