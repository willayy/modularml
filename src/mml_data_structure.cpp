#include "mml_data_structure.hpp"

template <typename T>
MML_DataStructure<T>::MML_DataStructure(const vec<int>& shape)
{
    if (shape.size() <= 0) throw std::invalid_argument("Shape is invalid");

    this->shape = shape;
    
    // Calculate the size using the diemensions provided
    int data_size = 1;
    for (int i : shape) data_size *= i;
    this->data = vec<T>(data_size);
}

template <typename T>
MML_DataStructure<T>::MML_DataStructure(const vec<int>& shape, const vec<T> data)
{
    if (shape.size() <= 0) throw std::invalid_argument("Shape isn't valid");

    int data_size = 1;
    for (int i : shape) data_size *= i;

    if (data_size != data.size()) throw std::invalid_argument("The size of the data and the desired shape does not match");

    this->shape = shape;
    this->data = data;
}

template <typename T>
T MML_DataStructure<T>::get(const vec<int>& indices) const
{
    if (indices.size() == 0) throw std::invalid_argument("Invalid indices");
    if (indices.size() != this->shape.size()) throw std::invalid_argument("The diemensions of the indices does not match with the data structure");
    
}

template <typename T>
void MML_DataStructure<T>::set(const vec<int>& indices, T value)
{
    
}

template <typename T>
void MML_DataStructure<T>::set_data(const vec<T> data)
{
    if (this->get_size() != data.size()) throw std::invalid_argument("The size of the data provided mismatch with the size of the data structure");
    
    for (int i : this->data.size())
    {
    }
}


