#pragma once

#include "a_node.hpp"
#include "a_tensor.hpp"
#include "globals.hpp"
#include "mml_arithmetic.hpp"

/**
 * @class LogSoftMaxNode
 * @brief A class representing a ReLU node in a computational graph.
 *
 * This class inherits from the Node class and represents the LogSoftMax node
 * in a computational graph. It performs the forward pass computation applying SoftMax and Logarithm along the specified axis.
 */
template <typename T>
class LogSoftMaxNode : public Node {
    static_assert(
        std::is_same_v<T, float> ||
        std::is_same_v<T, double>, 
        "LogSoftMaxNode supports only float, double"); // bfloat16 and float16 also but dont know how to include them currently

   public:
    using AbstractTensor = Tensor<T>;

    /**
     * @brief Constructor for LogSoftMaxNode.
     *
     * @param X Shared pointer to the tensor X.
     * @param Y Shared pointer to the output tensor.
     * @param axis Integer representing along which axis LogSoftMax is applied to. (default -1)
     */
    LogSoftMaxNode(shared_ptr<const AbstractTensor> X, shared_ptr<AbstractTensor> Y, int axis = -1);

    /**
     * @brief Perform the forward pass computation using LogSoftMax activation function.
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
     * @param inputs The input data to be set, where X is inputs[0].
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
    shared_ptr<const AbstractTensor> X;  // Input tensor X.
    shared_ptr<AbstractTensor> Y;        // Output tensor Y.
    int axis;
};

#include "../log_softmax_node.tpp"
