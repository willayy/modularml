#pragma once

#include "globals.hpp"
#include "a_model.hpp"

/**
 * @class DataParser
 * @brief Abstract class for parsing JSON data of a model into a Model Object.
 */
template <typename T>
class DataParser
{
  public:
    /**
     * @brief Parses JSON data of a model into a Model object.
     * 
     * @param data JSON data of a Model.
     */
    virtual std::unique_ptr<Model<T>> parse(const Data& data) const = 0;
    
    /// @brief Virtual destructor for cleanup.
    virtual ~DataParser() = default;
};