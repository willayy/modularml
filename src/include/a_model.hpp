#pragma once

#include "tensor.hpp"
#include "globals.hpp"

/**
 * @class Model
 * @brief Abstract base class for machine learning models.
 *
 * This class defines the interface for machine learning models, 
 * providing a method for inference and a virtual destructor.
 */
class Model {
public:
    /**
     * @brief Run the model with a single given input tensor.
     *
     * This is a pure virtual function that must be implemented by derived classes.
     *
     * @param inputs The input data for the model.
     * @return The result of the model inference.
     */
    virtual vector<GeneralDataTypes> infer(const vector<GeneralDataTypes>& inputs) = 0;

    /**
     * @brief Virtual destructor for the Model class.
     *
     * Ensures proper cleanup of derived class objects.
     */
    virtual ~Model() = default;
};