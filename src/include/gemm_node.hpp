#pragma once

#include "a_node.hpp"
#include "globals.hpp"

// Type constraints: no bfloat16 or float16 for now (not native to c++ 17).
using DataTypes = variant<
    Tensor_mml<float>,
    Tensor_mml<double>,
    Tensor_mml<int32_t>,
    Tensor_mml<int64_t>,
    Tensor_mml<uint32_t>,
    Tensor_mml<uint64_t>
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
             shared_ptr<DataTypes> Y,
             optional<shared_ptr<DataTypes>> C = std::nullopt,
             float alpha = 1.0f,
             float beta = 1.0f,
             int transA = 0,
             int transB = 0)
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
     * @param inputs The input data to be set, where A is inputs[0], B is inputs[1] and optionally C is inputs[2].
     */
    void setInputs(const vector<GeneralDataTypes>& inputs) override {
        if (inputs.size() < 2)
            throw runtime_error("GemmNode expects at least two inputs: A and B.");
    
        // Deduce type from the first input.
        visit([this, &inputs](const auto& tensorA) {
            using T = typename remove_reference_t<decltype(tensorA)>::value_type;
    
            // Restrict T to allowed types.
            if constexpr (!(std::is_same_v<T, float>   ||
                            std::is_same_v<T, double>  ||
                            std::is_same_v<T, int32_t> ||
                            std::is_same_v<T, int64_t> ||
                            std::is_same_v<T, uint32_t>||
                            std::is_same_v<T, uint64_t>))
            {
                throw runtime_error("GemmNode input type not supported.");
            } else {
                try {
                    auto valueA = std::get<Tensor_mml<T>>(inputs[0]);
                    auto valueB = std::get<Tensor_mml<T>>(inputs[1]);

                    A->template emplace<Tensor_mml<T>>(valueA);
                    B->template emplace<Tensor_mml<T>>(valueB);
                    
                    if (inputs.size() > 2) {
                        auto valueC = std::get<Tensor_mml<T>>(inputs[2]);
                        if (!C.has_value() || !C.value()) {
                            C = make_shared<DataTypes>(valueC);
                        } else {
                            C.value()->template emplace<Tensor_mml<T>>(valueC);
                        }
                    } else {
                        C.reset();
                    }
                } catch (const std::bad_variant_access&) {
                    throw runtime_error("Data type mismatch: All inputs must have the same type as A.");
                }
            }
        }, inputs[0]);
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