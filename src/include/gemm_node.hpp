#pragma once

#include "a_node.hpp"
#include "globals.hpp"

// Type constraints: no bfloat16 or float16 for now (not native to c++ 17).
using DataTypes = variant<
    Tensor<float>,
    Tensor<double>,
    Tensor<int32_t>,
    Tensor<int64_t>,
    Tensor<uint32_t>,
    Tensor<uint64_t>
>;

/**
 * @class GemmNode
 * @brief A class representing a GEMM node in a computational graph.
 *
 * This class inherits from the Node class and represents a General Matrix Multiply (GEMM) node
 * in a computational graph. It performs the forward pass computation using the GEMM inner product.
 */
class GemmNode : public Node {
public:
    /**
     * @brief Constructor for GemmNode.
     *
     * @param A Shared pointer to the tensor A.
     * @param B Shared pointer to the tensor B.
     * @param C Optional shared pointer to the tensor C.
     * @param Y Shared pointer to the output tensor.
     * @param alpha Scalar multiplier for A * B.
     * @param beta Scalar multiplier for C.
     * @param transA Whether to transpose A (0 means false).
     * @param transB Whether to transpose B (0 means false).
     */
    GemmNode(shared_ptr<DataTypes> A,
             shared_ptr<DataTypes> B,
             optional<shared_ptr<DataTypes>> C,
             shared_ptr<DataTypes> Y,
             float alpha,
             float beta,
             int transA,
             int transB)
      : A(A), B(B), C(C), Y(Y),
        alpha(alpha), beta(beta), transA(transA), transB(transB) {}

    /**
     * @brief Perform the forward pass computation using GEMM inner product.
     *
     * This function performs the forward pass computation using the General Matrix Multiply (GEMM) inner product.
     */
    void forward() override;
    
    /**
     * @brief Check if the input(s) are filled.
     * 
     * @return True if the input(s) are filled, false otherwise.
     */
    bool areInputsFilled() const override {
        return A && visit([](const auto& t) { return t.get_size() > 0; }, *A) &&
               B && visit([](const auto& t) { return t.get_size() > 0; }, *B) &&
               (!C.has_value() || (C.value() && visit([](const auto& t) { return t.get_size() > 0; }, *C.value())));
    }

    /**
     * @brief Set the input(s) for the node.
     * 
     * @param tensor The input data to be set.
     */
    void setInputs(const vector<GeneralDataTypes>& inputs) override {
        if (inputs.size() < 2) {
            throw std::runtime_error("GemmNode expects at least two inputs: A and B.");
        }
    
        // Set input A from index 0.
        A = make_shared<DataTypes>(
            visit([](const auto& t) -> DataTypes { return t; }, inputs[0])
        );
    
        // Set input B from index 1.
        B = make_shared<DataTypes>(
            visit([](const auto& t) -> DataTypes { return t; }, inputs[1])
        );
    
        // If a third input is provided, set input C from index 2.
        if (inputs.size() > 2) {
            C = make_shared<DataTypes>(
                visit([](const auto& t) -> DataTypes { return t; }, inputs[2])
            );
        } else {
            C.reset();  // or set to an empty optional if you prefer
        }
    }

    /**
     * @brief Check if the output(s) are filled.
     * 
     * @return True if the output(s) are filled, false otherwise.
     */
    bool areOutputsFilled() const override {
        return Y && visit([](const auto& t) { return t.get_size() > 0; }, *Y);
    }

    /**
     * @brief Get the output of the node.
     * 
     * @return The output data.
     */
    vector<GeneralDataTypes> getOutputs() const override {
        if (!Y) {
            throw runtime_error("Output tensor Y is not filled!");
        }
        return { visit([](const auto& arg) -> GeneralDataTypes { return arg; }, *Y) };
    }

private:
    // Inputs
    shared_ptr<DataTypes> A; // Input tensor A.
    shared_ptr<DataTypes> B; // Input tensor B.
    optional<shared_ptr<DataTypes>> C; // Optional tensor C.

    // Output
    shared_ptr<DataTypes> Y; // Output tensor.

    // Attributes
    float alpha;  // Scalar multiplier for A * B.
    float beta;   // Scalar multiplier for C.
    int transA;   // Whether to transpose A (0: no, non-zero: yes).
    int transB;   // Whether to transpose B (0: no, non-zero: yes).
};