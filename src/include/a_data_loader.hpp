#pragma once

#include "globals.hpp"
#include "a_tensor.hpp"

/**
 * @class DataLoader
 * @brief Abstract base class for machine learning models.
 *
 * This class defines the interface for loading exeternal data and translating it to a Tensor, 
 * Provides a method for loading data and a virtual destructor.
 * 
 * @author Tim Carlsson (timca@chalmers.se)
 */

// The use of templates in this class is not prefered for the dataloader, is there a better approach?
template <typename T>
class DataLoader {
public:
    /**
     * @brief Load external data and translate it to tensor.
     *
     * This is a pure virtual function that must be implemented by derived classes.
     *
     * @return A unique_ptr to a Tensor containing the data that was loaded.
     */
    virtual unique_ptr<Tensor<T>> load() const = 0;

    /**
     * @brief Virtual destructor for the DataLoader class.
     *
     * Ensures proper cleanup of derived class objects.
     */
    virtual ~DataLoader() = default;
};