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
    FlattenNode(shared_ptr<AbstractTensor> X,
                shared_ptr<AbstractTensor> Y,
                int axis = 1);


    /**
     * @brief
     */
    void forward() override;

    /**
     * @brief Check if the input(s) are filled.
     *
     * @return True if the input(s) are filled, false otherwise.
     */
    bool areInputsFilled() const override;

    /**
     * @brief Set the input(s) for the node.
     *
     * @param inputs The input data to be set.
     */
    void setInputs(const array_mml<GeneralDataTypes>& inputs) override;

    /**
     * @brief Check if the output(s) are filled.
     *
     * @return True if the output(s) are filled, false otherwise.
     */
    bool areOutputsFilled() const override;

    /**
     * @brief Get the output of the node.
     *
     * @return The output data.
     */
    array_mml<GeneralDataTypes> getOutputs() const override;

private:
    shared_ptr<Tensor<T>> X;
    shared_ptr<Tensor<T>> Y;
    int axis;


    int get_axis() const;
};

#include "../flatten_node.tpp"