#include "flatten_node.hpp"

template <typename T>
FlattenNode<T>::FlattenNode(shared_ptr<AbstractTensor> X,
                            shared_ptr<AbstractTensor> Y,
                            int axis) : X(X), Y(Y), axis(axis) {}


template <typename T>
void FlattenNode<T>::forward() {
    auto input_copy = X->copy();

    int height_2d, width_2d;

    if (get_axis() == 0) {
        // This gives a warning, but when get_size returns int in the future it will disappear
        input_copy->reshape({input_copy->get_size()});
    }

    else { 
        height_2d = 1;
        width_2d = 1;
        
        int i = 0;
        for (i; i < axis; i++) {
            height_2d *= input_copy->get_shape()[i];
        }
        for (i; i < input_copy->get_shape().size(); i++) {
            width_2d *= input_copy->get_shape()[i];
        }
    }

    input_copy->reshape({height_2d, width_2d});

    *Y = *input_copy;
}

template <typename T>
bool FlattenNode<T>::areInputsFilled() const {
    return (X && X->get_size() > 0);
}

template <typename T>
void FlattenNode<T>::setInputs(const array_mml<GeneralDataTypes>& inputs) {
    if (inputs.size() == 1) {
        auto x_value = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
        *X = *x_value;
    }
    else {
        throw invalid_argument("Invalid input array");
    }
}

template <typename T>
bool FlattenNode<T>::areOutputsFilled() const {
    return Y && Y->get_size() > 0;
}

template <typename T>
array_mml<GeneralDataTypes> FlattenNode<T>::getOutputs() const {
    return array_mml<GeneralDataTypes>{GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y))};
}

template <typename T>
int FlattenNode<T>::get_axis() const {
    return axis;
}