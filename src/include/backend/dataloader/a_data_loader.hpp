#pragma once

#include "globals.hpp"
#include "datastructures/a_tensor.hpp"
#include "backend/dataloader/data_loader_config.hpp"

/**
 * @class DataLoader
 * @brief Abstract base class for machine learning models.
 *
 * This class defines the interface for loading exeternal data and translating it to a Tensor, 
 * Provides a method for loading data and a virtual destructor.
 * 
 * @author Tim Carlsson (timca@chalmers.se)
 */

template <typename T>
class DataLoader {
public:

    DataLoader() = default;
    
    /**
     * @brief Load external data and translate it to tensor.
     * 
     * This is a pure virtual function that must be implemented by derived classes.
     *
     * @return A unique_ptr to a Tensor containing the data.
     */
    virtual std::shared_ptr<Tensor<T>> load(const DataLoaderConfig& config) const = 0;

    /**
     * @brief Virtual destructor for the DataLoader class.
     *
     * Ensures proper cleanup of derived class objects.
     */
    virtual ~DataLoader() = default;
};