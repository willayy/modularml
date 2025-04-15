#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>  // IWYU pragma: keep

#include "datastructures/a_tensor.hpp"

#define ASSERT_ALLOWED_TYPES_ONNX_GM(T)  \
  static_assert(std::is_arithmetic_v<T>, \
                "Data structure type must be an arithmetic type.")

/// @brief Abstract class for classes that contain standard GEMM functions using
/// the ONNX GEMM format.
/// @details The difference between the ONNX GEMM and standard GEMM.
/// GEMM: C := alpha * op( A ) * op( B ) + beta * C
/// ONNX GEMM: Y := alpha * A * B + beta * C (std::optional)
/// @tparam T the type of the data that the GEMM functions will operate on.
template <typename T>
class OnnxGemmModule {
 public:
  /// @brief Default constructor for GEMM class.
  [[deprecated("Use TensorOperationsModule instead")]]
  OnnxGemmModule() = default;

  /// @brief Copy constructor for GEMM class.
  [[deprecated("Use TensorOperationsModule instead")]]
  OnnxGemmModule(const OnnxGemmModule &other) = default;

  /// @brief Move constructor for GEMM class.
  [[deprecated("Use TensorOperationsModule instead")]]
  OnnxGemmModule(OnnxGemmModule &&other) noexcept = default;

  /// @brief Abstract destructor for GEMM class.
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual ~OnnxGemmModule() = default;

  /**
   * @brief Basic CPU implementation of GEMM, with the inner product approach.
   * Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_inner_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the outer product approach.
   * Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_outer_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the row-wise product
   * approach. Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_row_wise_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the col-wise product
   * approach. Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_col_wise_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;

  /**
   * @brief Blocked CPU implementation of GEMM.
   * Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_blocked(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;

  /**
   * @brief Vectorized implementation of GEMM using SIMD thanks to AVX.
   * Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_avx(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;

  /**
   * @brief Vectorized implementation of GEMM using SIMD thanks to AVX512.
   * Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_avx512(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;

  /**
   * @brief GEMM using Intel's Math Kernel Library.
   * Performs operation Y := alpha * A * B + beta * C (std::optional)
   * More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0, or
   * (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0, or
   * (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is transposed.
   * @param C Optional input tensor C. If not specified, the computation is done
   * as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N).
   */
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual std::shared_ptr<Tensor<T>> gemm_intel_MKL(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) = 0;
};