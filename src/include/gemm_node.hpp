#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_tensor.hpp"
#include "mml_gemm.hpp"

/**
 * @class GemmNode
 * @brief A class representing a GEMM node in a computational graph.
 *
 * This class inherits from the Node class and represents a General Matrix Multiply (GEMM) node
 * in a computational graph. It performs the forward pass computation using the GEMM inner product.
 */
template <typename T>
class GemmNode : public Node {
    static_assert(
        std::is_same_v<T, float>   ||
        std::is_same_v<T, double>  ||
        std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t> ||
        std::is_same_v<T, uint32_t>||
        std::is_same_v<T, uint64_t>,
        "GemmNode_T supports only float, double, int32_t, int64_t, uint32_t, or uint64_t");
public:
    using AbstractTensor = Tensor<T>;

    /**
     * @brief Constructor for GemmNode.
     *
     * @param A Shared pointer to the tensor A.
     * @param B Shared pointer to the tensor B.
     * @param Y Shared pointer to the output tensor.
     * @param C Optional shared pointer to the tensor C.
     * @param alpha Scalar multiplier for A * B.
     * @param beta Scalar multiplier for C.
     * @param transA Whether to transpose A (0 means false).
     * @param transB Whether to transpose B (0 means false).
     */
    GemmNode(shared_ptr<AbstractTensor> A,
             shared_ptr<AbstractTensor> B,
             shared_ptr<AbstractTensor> Y,
             optional<shared_ptr<AbstractTensor>> C = std::nullopt,
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
    void forward() override {
        if (!areInputsFilled())
            throw runtime_error("GemmNode inputs are not fully set.");

       
        auto shapeA = A->get_shape();
        if (shapeA.size() < 2)
            throw runtime_error("Tensor A must be at least 2D.");

        int M = shapeA[0];  // Number of rows.
        int K = shapeA[1];  // Number of columns of A.

        
        auto shapeB = B->get_shape();
        if (shapeB.size() < 2)
            throw runtime_error("Tensor B must be at least 2D.");
        if (shapeB[0] != K)
            throw runtime_error("GemmNode: Dimension mismatch between A and B.");
        
        int N = shapeB[1];  // Number of columns of B.
        
        int lda = K;
        int ldb = N;
        int ldc = N;

        // Handling optional C tensor not implemented directly in gemm_inner_product. 
        // Will have to be done here instead by creating suboptimal concrete tensor.
        // Gemm_inner_product could be modified to handle optional C tensor and take output Y.
        shared_ptr<Tensor_mml<T>> C_ptr;
        if (C.has_value() && C.value()) {
            C_ptr = std::dynamic_pointer_cast<Tensor_mml<T>>(C.value());
            if (!C_ptr)
                throw runtime_error("GemmNode: Failed to cast optional C to Tensor_mml<T>.");
        } else {
            Tensor_mml<T> zero_tensor({M, N});
            zero_tensor.fill(static_cast<T>(0));
            C_ptr = make_shared<Tensor_mml<T>>(zero_tensor);
        }

        Gemm_mml<T> gemm;
        gemm.gemm_inner_product(0, 0, M, N, K, static_cast<T>(alpha),
                                A, lda,
                                B, ldb,
                                static_cast<T>(beta),
                                C_ptr, ldc);

        if (!Y)
            throw runtime_error("Output tensor Y is not allocated.");

        auto y_mml = std::static_pointer_cast<Tensor_mml<T>>(Y);
        auto c_mml = std::static_pointer_cast<Tensor_mml<T>>(C_ptr);
        y_mml->update_from(*c_mml);
    };
    
    /**
     * @brief Check if the input(s) are filled.
     * 
     * @return True if the input(s) are filled, false otherwise.
     */
    bool areInputsFilled() const override {
        return A && A->get_size() > 0 &&
               B && B->get_size() > 0 &&
               (!C.has_value() || (C.value() && C.value()->get_size() > 0));
    }

    /**
     * @brief Set the input(s) for the node.
     * 
     * @param inputs The input data to be set, where A is inputs[0], B is inputs[1] and optionally C is inputs[2].
     */
    void setInputs(const array_mml<GeneralDataTypes>& inputs) override {
        if (inputs.size() < 2)
            throw runtime_error("GemmNode expects at least two inputs: A and B.");

        auto valueA = std::get<std::shared_ptr<AbstractTensor>>(inputs[0]);
        auto valueB = std::get<std::shared_ptr<AbstractTensor>>(inputs[1]);
    
        // Update A in place using update_from
        auto a_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(A);
        auto valueA_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueA);
        if (!a_mml || !valueA_mml)
            throw std::runtime_error("Failed to cast A or input A to Tensor_mml<T>.");
        a_mml->update_from(*valueA_mml);
    
        // Update B in place using update_from
        auto b_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(B);
        auto valueB_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueB);
        if (!b_mml || !valueB_mml)
            throw std::runtime_error("Failed to cast B or input B to Tensor_mml<T>.");
        b_mml->update_from(*valueB_mml);
    
        // Handle optional C.
        if (inputs.size() > 2) {
            auto valueC = std::get<std::shared_ptr<AbstractTensor>>(inputs[2]);
            if (!C.has_value() || !C.value()) {
                C = valueC;
            } else {
                auto c_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(C.value());
                auto valueC_mml = std::dynamic_pointer_cast<Tensor_mml<T>>(valueC);
                if (!c_mml || !valueC_mml)
                    throw std::runtime_error("Failed to cast C or input C to Tensor_mml<T>.");
                c_mml->update_from(*valueC_mml);
            }
        } else {
            C.reset();
        }
    }

    /**
     * @brief Check if the output(s) are filled.
     * 
     * @return True if the output(s) are filled, false otherwise.
     */
    bool areOutputsFilled() const override {
        return Y && Y->get_size() > 0;
    }

    /**
     * @brief Get the output of the node.
     * 
     * @return The output data.
     */
    array_mml<GeneralDataTypes> getOutputs() const override {
        return array_mml<GeneralDataTypes>{ GeneralDataTypes(std::static_pointer_cast<AbstractTensor>(Y)) };
    }

private:
    // Inputs
    shared_ptr<AbstractTensor> A; // Input tensor A.
    shared_ptr<AbstractTensor> B; // Input tensor B.
    optional<shared_ptr<AbstractTensor>> C; // Optional tensor C.

    // Output
    shared_ptr<AbstractTensor> Y; // Output tensor.

    // Attributes
    float alpha;  // Scalar multiplier for A * B.
    float beta;   // Scalar multiplier for C.
    int transA;   // Whether to transpose A (0: no, non-zero: yes).
    int transB;   // Whether to transpose B (0: no, non-zero: yes).
};