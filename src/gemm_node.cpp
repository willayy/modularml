#include "include/gemm_node.hpp"
#include "include/mml_gemm.hpp"

void GemmNode::forward() {
    if (!areInputsFilled())
        throw std::runtime_error("GemmNode inputs are not fully set.");

    // Use std::visit to dispatch based on the concrete type stored in A.
    std::visit([this](auto &a_tensor) {
        // Extract the scalar type T from the tensor.
        using T = typename decltype(a_tensor)::value_type; // Requires Tensor to expose value_type

        // Retrieve shape information using the public getter.
        auto shapeA = a_tensor.get_shape(); // Expecting shapeA to be an array_mml<int>
        if (shapeA.size() < 2)
            throw std::runtime_error("Tensor A must be at least 2D.");

        int M = shapeA[0];  // Number of rows.
        int K = shapeA[1];  // Number of columns of A.

        // Get tensor B (and verify its shape).
        auto &b_tensor = std::get<T>(*B);
        auto shapeB = b_tensor.get_shape();
        if (shapeB.size() < 2)
            throw std::runtime_error("Tensor B must be at least 2D.");
        if (shapeB[0] != K)
            throw std::runtime_error("GemmNode: Dimension mismatch between A and B.");
        int N = shapeB[1];  // Number of columns of B.

        // Create an output tensor of shape [M, N] via the public interface.
        Tensor<T> y_tensor({M, N});
        y_tensor.fill(static_cast<T>(0));  // Initialize to zero.

        // Wrap A and B into shared_ptrs.
        auto A_ptr = std::make_shared<Tensor<T>>(a_tensor);
        auto B_ptr = std::make_shared<Tensor<T>>(b_tensor);

        // Prepare C: if provided, wrap it; otherwise, create a zero tensor.
        std::shared_ptr<Tensor<T>> C_ptr;
        if (C.has_value() && C.value()) {
            auto &c_tensor = std::get<T>(*C.value());
            C_ptr = std::make_shared<Tensor<T>>(c_tensor);
        } else {
            Tensor<T> zero_tensor({M, N});
            zero_tensor.fill(static_cast<T>(0));
            C_ptr = std::make_shared<Tensor<T>>(zero_tensor);
        }

        // (Y_ptr isn't strictly needed since we assign the result back to Y later.)
        auto Y_ptr = std::make_shared<Tensor<T>>(y_tensor);

        // Set up leading dimensions assuming row-major storage.
        int lda = K;  // A is M x K.
        int ldb = N;  // B is K x N.
        int ldc = N;  // C and Y are M x N.

        // Create a GEMM module instance and perform the inner product.
        Gemm_mml<T> gemm;
        gemm.gemm_inner_product(0, 0, M, N, K, static_cast<T>(alpha),
                                  A_ptr, lda,
                                  B_ptr, ldb,
                                  static_cast<T>(beta),
                                  C_ptr, ldc);

        // Store the result in Y.
        *Y = *C_ptr;
    }, *A);
}