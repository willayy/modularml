#pragma once

#include "a_tensor.hpp"
#include "a_layer.hpp"
#include <vector>


/**
 * @class Model
 * @brief Abstract Model class/type.
 *
 * This class provides an interface for models
 */

/// @brief Abstract Model class/type.

template <typename T>
class Model {

protected:

    /// @brief Dynamic array of Layers
    std::vector<Layer> layers;

public:
    /// @brief Default constructor for Model.
    explicit Model() = default;

    /// @brief Virtual destructor for Model.
    /// @details Ensures derived class destructors are called properly.
    virtual ~Model() = default;


    /// @brief Make an inference.
    /// @param t Input data
    /// @return Predicted data
    virtual Tensor<T> infer(const Tensor<T>& t) const = 0;

};
