#pragma once

#include "a_data_structure.hpp"

template <typename T>
class MML_DataStructure : public DataStructure<T>
{ 
  public:

    /**
     * @brief Constructor for the MML_DataStructure class
     * @param shape A vector defining the diemensions of the data
     */
    MML_DataStructure(const vec<int>& shape);

    /**
     * @brief Constructor for the MML_DataStructure class
     * @param shape A vector defining the diemensions of the data
     * @param data A vector containing the data for the structure
     */
    MML_DataStructure(const vec<int>& shape, const vec<T> data);

    void set_data(const vec<T> data) override;
    
    void set_zero() override;

    const vec<int> &get_shape() const override;

    const string get_shape_str() const override;
    
    int get_size() const override;
    
    bool equals(const MML_DataStructure<T> &other) const override;
    
    T get(const vec<int> &indices) const override;

    void set(const vec<int> &indices, T value) override;


  private:
    std::vector<int> shape;
    std::vector<T> data;

    /**
     * @brief Given the indices and the shape of the data structure calculates the offset that the indices represent.
     * @param indices A vector of integers that each represent a direction in a diemension.
     */
    int calc_offset(const vec<int> &indices) const;
    
}; 