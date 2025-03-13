#include "flatten_node.hpp"

template <typename T>
FlattenNode<T>::FlattenNode(shared_ptr<AbstractTensor> X,
                            shared_ptr <AbstractTensor> Y,
                            int axis = 1)
    : X(X), Y(Y), axis(axis) {}


template <typename T>
void FlattenNode<T>::forward() {

    auto input_copy = X->copy();


}