#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_tensor.hpp"

template <typename T>
class FlattenNode : public Node {
public:
    using AbstractTensor = Tensor<T>;

    /**
     * @brief Constructor for FlattenNode
     */
    FlattenNode(shared_ptr)


private:
    shared_ptr<Tensor<T>> input;
    shared_ptr<Tensor<T>> output;
    int axis;
};