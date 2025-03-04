#pragma once

#include "tensor.hpp"
#include "globals.hpp"

// More data types can be added as needed
using GeneralDataTypes = variant<Tensor<int>, Tensor<float>, Tensor<double>>;

/**
 * @class Node
 * @brief Abstract base class representing a node in a computational graph.
 *
 * This class defines the interface for nodes in a computational graph, 
 * including methods for forward propagation, input/output management, 
 * and checking the status of inputs and outputs.
 */
class Node {
public:
    /**
     * @brief Perform the forward pass computation.
     *
     * This pure virtual function must be overridden by derived classes to 
     * implement the specific forward pass logic. It modifies the output(s) 
     * in place.
     */
    virtual void forward() = 0;

    /**
     * @brief Check if the input(s) are filled.
     *
     * This pure virtual function must be overridden by derived classes to 
     * determine if the input(s) required for computation are filled.
     * 
     * @return True if the input(s) are filled, false otherwise.
     */
    virtual bool areInputsFilled() const = 0;

    /**
     * @brief Set the input(s) for the node.
     *
     * This pure virtual function must be overridden by derived classes to 
     * set the input(s) for the node. Currently, it supports a singular input 
     * as the graph runs with a singular input.
     * 
     * @param tensor The input data to be set.
     */
    virtual void setInput(const GeneralDataTypes& tensor) = 0;

    /**
     * @brief Check if the output(s) are filled.
     *
     * This pure virtual function must be overridden by derived classes to 
     * determine if the output(s) of the node are filled.
     * 
     * @return True if the output(s) are filled, false otherwise.
     */
    virtual bool areOutputsFilled() const = 0;

    /**
     * @brief Get the output of the node.
     *
     * This pure virtual function must be overridden by derived classes to 
     * retrieve the output of the node. Currently, it supports a singular output.
     * 
     * @return The output data.
     */
    virtual GeneralDataTypes getOutput() const = 0;

    /**
     * @brief Virtual destructor for the Node class.
     *
     * Ensures derived class destructors are called properly.
     */
    virtual ~Node() = default;
};
