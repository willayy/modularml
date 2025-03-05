#pragma once

#include "tensor.hpp"
#include "globals.hpp"

// Type constraints: no bfloat16 or float16 for now (not native to c++ 17). Also maybe exists more don't know.
using GeneralDataTypes = variant<
    Tensor<bool>,
    Tensor<double>,
    Tensor<float>,
    Tensor<int16_t>,
    Tensor<int32_t>,
    Tensor<int64_t>,
    Tensor<int8_t>,
    Tensor<uint16_t>,
    Tensor<uint32_t>,
    Tensor<uint64_t>,
    Tensor<uint8_t>
>;

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
     * set the input(s) for the node.
     * 
     * @param inputs The input data to be set.
     */
    virtual void setInputs(const vector<GeneralDataTypes>& inputs) = 0; // This function could have better type safety somehow maybe.

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
    virtual vector<GeneralDataTypes> getOutputs() const = 0;

    /**
     * @brief Virtual destructor for the Node class.
     *
     * Ensures derived class destructors are called properly.
     */
    virtual ~Node() = default;
};
