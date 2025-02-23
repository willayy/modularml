#include "mml_data_structure.hpp"

template <typename T>
MML_DataStructure<T>::MML_DataStructure(const vec<int>& shape)
{
    assert(shape.size() > 0); // Check that the shape is valid
    this->shape = shape;
    
    // Calculate the size using the diemensions provided
    int data_size = 1;
    for (int i : shape) data_size *= i;
    this->data = vec<T>(data_size);
}

template <typename T>
MML_DataStructure<T>::MML_DataStructure(const vec<int>& shape, const vec<T> data)
{
    assert(shape.size() > 0); // Check that the shape is valid
    int data_size = 1;
    for (int i : shape) data_size *= i;

    assert(data_size == data.size());

    this->shape = shape;
    this->data = data;
}


