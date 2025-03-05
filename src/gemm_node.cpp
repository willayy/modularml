#include "include/gemm_node.hpp"
#include "include/mml_gemm.hpp"
#include "include/mml_tensor.hpp"

void GemmNode::forward() {
    if (!areInputsFilled())
        throw std::runtime_error("GemmNode inputs are not fully set.");

    std::visit([this](auto &a_tensor) {
        using T = typename std::remove_reference_t<decltype(a_tensor)>::value_type;

        // Retrieve shape of A using the public API.
        auto shapeA = a_tensor.get_shape();
        if (shapeA.size() < 2)
            throw std::runtime_error("Tensor A must be at least 2D.");

        int M = shapeA[0];  // Number of rows.
        int K = shapeA[1];  // Number of columns of A.

        auto &b_tensor = std::get<Tensor_mml<T>>(*B);
        auto shapeB = b_tensor.get_shape();
        if (shapeB.size() < 2)
            throw std::runtime_error("Tensor B must be at least 2D.");
        if (shapeB[0] != K)
            throw std::runtime_error("GemmNode: Dimension mismatch between A and B.");
        int N = shapeB[1];  // Number of columns of B.

        auto A_ptr = std::make_shared<Tensor_mml<T>>(a_tensor);
        auto B_ptr = std::make_shared<Tensor_mml<T>>(b_tensor);

        // Prepare C: if provided, wrap it; otherwise, create a zero tensor.
        std::shared_ptr<Tensor_mml<T>> C_ptr;
        if (C.has_value() && C.value()) {
            auto &c_tensor = std::get<Tensor_mml<T>>(*C.value());
            C_ptr = std::make_shared<Tensor_mml<T>>(c_tensor);
        } else {
            Tensor_mml<T> zero_tensor({M, N});
            zero_tensor.fill(static_cast<T>(0));
            C_ptr = std::make_shared<Tensor_mml<T>>(zero_tensor);
        }
        
        int lda = K;
        int ldb = N;
        int ldc = N;

        Gemm_mml<T> gemm;
        gemm.gemm_inner_product(0, 0, M, N, K, static_cast<T>(alpha),
                                  A_ptr, lda,
                                  B_ptr, ldb,
                                  static_cast<T>(beta),
                                  C_ptr, ldc);

        // Update the output tensor.
        Y->template emplace<Tensor_mml<T>>(*C_ptr);
    }, *A);
}