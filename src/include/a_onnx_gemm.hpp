#pragma once

#include "a_tensor.hpp"
#include "globals.hpp"

#define ASSERT_ALLOWED_TYPES_ONNX_GM(T) static_assert(std::is_arithmetic_v<T>, "Data structure type must be an arithmetic type.")

/// @brief Abstract class for classes that contain standard GEMM functions using the ONNX GEMM format.
/// @details The differnnce between the ONNX GEMM and standard GEMM.
/// GEMM: C := alpha * op( A ) * op( B ) + beta * C
/// ONNX GEMM: Y := alpha * A * B + beta * C (optional)
/// @tparam T the type of the data that the GEMM functions will operate on.
template <typename T>
class OnnxGemmModule {
 public:
  /// @brief Default constructor for GEMM class.
  OnnxGemmModule() = default;

  /// @brief Copy constructor for GEMM class.
  OnnxGemmModule(const OnnxGemmModule& other) = default;

  /// @brief Move constructor for GEMM class.
  OnnxGemmModule(OnnxGemmModule&& other) noexcept = default;

  /// @brief Abstract destructor for GEMM class.
  virtual ~OnnxGemmModule() = default;

  /**
   * @brief Basic CPU implementation of GEMM, with the inner product approach.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @author William Norland
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_inner_product(float alpha = 1.0, float beta = 1.0,
                                                   int transA = 0, int transB = 0,
                                                   shared_ptr<Tensor<T>> A,
                                                   shared_ptr<Tensor<T>> B,
                                                   optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the outer product approach.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA
   * @param transB
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_outer_product(float alpha = 1.0, float beta = 1.0,
                                                   int transA = 0, int transB = 0,
                                                   shared_ptr<Tensor<T>> A,
                                                   shared_ptr<Tensor<T>> B,
                                                   optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the row-wise product approach.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_row_wise_product(float alpha = 1.0, float beta = 1.0,
                                                      int transA = 0, int transB = 0,
                                                      shared_ptr<Tensor<T>> A,
                                                      shared_ptr<Tensor<T>> B,
                                                      optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the col-wise product approach.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_col_wise_product(float alpha = 1.0, float beta = 1.0,
                                                      int transA = 0, int transB = 0,
                                                      shared_ptr<Tensor<T>> A,
                                                      shared_ptr<Tensor<T>> B,
                                                      optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief Blocked CPU implementation of GEMM.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_blocked(float alpha = 1.0, float beta = 1.0,
                                             int transA = 0, int transB = 0,
                                             shared_ptr<Tensor<T>> A,
                                             shared_ptr<Tensor<T>> B,
                                             optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief Vectorized implementation of GEMM using SIMD thanks to AVX.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_avx(float alpha = 1.0, float beta = 1.0,
                                         int transA = 0, int transB = 0,
                                         shared_ptr<Tensor<T>> A,
                                         shared_ptr<Tensor<T>> B,
                                         optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief Vectorized implementation of GEMM using SIMD thanks to AVX512.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_avx512(float alpha = 1.0, float beta = 1.0,
                                            int transA = 0, int transB = 0,
                                            shared_ptr<Tensor<T>> A,
                                            shared_ptr<Tensor<T>> B,
                                            optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief GEMM using Intel's Math Kernel Library.
   * Performs operation Y := alpha * A * B + beta * C (optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
   * @param C Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  virtual shared_ptr<Tensor<T>> gemm_intel_MKL(float alpha = 1.0, float beta = 1.0,
                                               int transA = 0, int transB = 0,
                                               shared_ptr<Tensor<T>> A,
                                               shared_ptr<Tensor<T>> B,
                                               optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;
};