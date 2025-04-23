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

#include "a_tensor.hpp"
#include "tensor_default_operation_functions.hpp"
#include "tensor_operation_function_types.hpp"
#include "tensor_concept.hpp"

/**
 * A module for performing arithmetic operations on tensor structures. Your
 * static modular toolbox for all operations on tensors.
 */
class TensorOperations {
 public:
  TensorOperations(const TensorOperations&) = delete;
  TensorOperations& operator=(const TensorOperations&) = delete;

  /**
   * @brief General matrix multiplication (GEMM) std::function.
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://www.netlib.org/blas/
   * @author Mateo Vazquez Maceiras (maceiras@chalmers.se)
   * @author William Norland (C++ implementation)
   * @param TA True if matrix A is transposed.
   * @param TB True if matrix B is transposed.
   * @param M Number of rows in matrix A and C.
   * @param N Number of columns in matrix B and C.
   * @param K Number of columns in matrix A and rows in matrix B.
   * @param ALPHA Scalar alpha.
   * @param A 1D array containing the first matrix.
   * @param lda Specifies the first dimension of matrix A.
   * @param B 1D array containing the second matrix.
   * @param ldb Specifies the first dimension of matrix B.
   * @param BETA Scalar beta.
   * @param C 1D array containing the result matrix (can be initialized to
   * non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C. */
  template <TensorConcept::Types T>
  static void gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                   std::shared_ptr<Tensor<T>> A, int lda,
                   std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                   std::shared_ptr<Tensor<T>> C, int ldc);

  /**
   * @brief Sets the gemm std::function pointer.
   * @param ptr Function pointer to the gemm implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_gemm_ptr(toft::gemm_func<Ts>... ptr);

  /**
   * @brief General matrix multiplication (GEMM) std::function using the ONNX
   * standard. Performs operation Y := alpha * A * B + beta * C
   * (std::optional) More detailed info in
   * https://onnx.ai/onnx/operators/onnx__Gemm.html
   * @param A Input tensor A. The shape of A should be (M, K) if transA is 0,
   * or (K, M) if transA is non-zero.
   * @param B Input tensor B. The shape of B should be (K, N) if transB is 0,
   * or (N, K) if transB is non-zero.
   * @param alpha Float alpha.
   * @param beta Float beta.
   * @param transA 0 if matrix A is not transposed, 1 if matrix A is
   * transposed.
   * @param transB 0 if matrix B is not transposed, 1 if matrix B is
   * transposed.
   * @param C Optional input tensor C. If not specified, the computation is
   * done as if C is a scalar 0. The shape of C should be unidirectional
   * broadcastable to (M, N).
   * @return Output tensor Y. Output tensor of shape (M, N). */
  template <TensorConcept::Types T>
  static std::shared_ptr<Tensor<T>> gemm_onnx(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt);

  /**
   * @brief Sets the gemm_onnx std::function pointer.
   * @param ptr Function pointer to the gemm_onnx implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_gemm_onnx_ptr(toft::gemm_onnx_func<Ts>... ptr);

  /**
   * @brief Adds two tensors element-wise.
   * Performs operation c = a + b.
   * @param a First tensor.
   * @param b Second tensor.
   * @param c Output tensor. */
  template <TensorConcept::Types T>
  static void add(const std::shared_ptr<const Tensor<T>> a,
                  const std::shared_ptr<const Tensor<T>> b,
                  std::shared_ptr<Tensor<T>> c);

  /**
   * @brief Sets the add std::function pointer.
   * @param ptr Function pointer to the add implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_add_ptr(toft::add_func<Ts>... ptr);

  /**
   * @brief Subtracts two tensors element-wise.
   * Performs operation c = a - b.
   * @param a First tensor.
   * @param b Second tensor.
   * @param c Output tensor. */
  template <TensorConcept::Types T>
  static void subtract(const std::shared_ptr<Tensor<T>> a,
                       const std::shared_ptr<Tensor<T>> b,
                       std::shared_ptr<Tensor<T>> c);

  /**
   * @brief Sets the subtract std::function pointer.
   * @param ptr Function pointer to the subtract implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_subtract_ptr(toft::subtract_func<Ts>... ptr);

  /**
   * @brief Multiplies a tensor by a scalar.
   * Performs operation c = a * b.
   * @param a Input tensor.
   * @param b Scalar.
   * @param c Output tensor. */
  template <TensorConcept::Types T>
  static void multiply(const std::shared_ptr<Tensor<T>> a, const T b,
                       std::shared_ptr<Tensor<T>> c);

  /**
   * @brief Sets the multiply std::function pointer.
   * @param ptr Function pointer to the multiply implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_multiply_ptr(toft::multiply_func<Ts>... ptr);

  /**
   * @brief Compares two tensors element-wise.
   * @param a First tensor.
   * @param b Second tensor.
   * @return True if the tensors are std::equal, false otherwise. */
  template <TensorConcept::Types T>
  static bool equals(const std::shared_ptr<Tensor<T>> a,
                     const std::shared_ptr<Tensor<T>> b);

  /**
   * @brief Sets the equals std::function pointer.
   * @param ptr Function pointer to the equals implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_equals_ptr(toft::equals_func<Ts>... ptr);

  /**
   * @brief Applies a std::function element-wise to a tensor.
   * @param a Input tensor.
   * @param f Function to apply.
   * @param c Output tensor. */
  template <TensorConcept::Types T>
  static void elementwise(const std::shared_ptr<const Tensor<T>> a,
                          const std::function<T(T)> f,
                          const std::shared_ptr<Tensor<T>> c);

  /**
   * @brief Sets the elementwise std::function pointer.
   * @param ptr Function pointer to the elementwise implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_elementwise_ptr(toft::elementwise_func<Ts>... ptr);

  /**
   * @brief Applies a std::function element-wise to a tensor in place.
   * @param a Input tensor.
   * @param f Function to apply. */
  template <TensorConcept::Types T>
  static void elementwise_in_place(const std::shared_ptr<Tensor<T>> a,
                                   const std::function<T(T)> f);

  /**
   * @brief Sets the elementwise_in_place std::function pointer.
   * @param ptr Function pointer to the elementwise_in_place implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_elementwise_in_place_ptr(
      toft::elementwise_in_place_func<Ts>... ptr);

  /**
   * @brief Gets the maximum value of a tensor along a given axis.
   * @param a Input tensor.
   * @return The maximum value along the specified axis.
   */
  template <TensorConcept::Types T>
  static int arg_max(const std::shared_ptr<const Tensor<T>> a);

  /**
   * @brief Sets the arg_max std::function pointer.
   * @param ptr Function pointer to the arg_max implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_arg_max_ptr(toft::arg_max_func<Ts>... ptr);

  /**
   * @brief Applies a sliding window operation to a tensor.
   * @param in_shape Input shape.
   * @param out_shape Output shape.
   * @param kernel_shape Shape of the kernel.
   * @param strides Strides for the sliding window.
   * @param dilations Dilations for the sliding window.
   * @param pads Padding for the sliding window.
   * @param window_f Function to apply in the sliding window.
   */
  template <TensorConcept::Types T>
  static void sliding_window(
      const array_mml<size_t>& in_shape, const array_mml<size_t>& out_shape,
      const std::vector<int>& kernel_shape, const std::vector<int>& strides,
      const std::vector<int>& dilations,
      const std::vector<std::pair<int, int>>& pads,
      const std::function<void(const std::vector<std::vector<size_t>>&,
                               const std::vector<size_t>&)>& window_f);

  /**
   * @brief Sets the sliding_window function pointer.
   * @param ptr Function pointer to the sliding_window implementation.
   */
  template <TensorConcept::Types... Ts>
  static void set_sliding_window_ptr(toft::sliding_window_func<Ts>... ptr);

 private:
  // Private constructor.
  TensorOperations() = default;

  // Pointer to the sliding_window function.
  template <TensorConcept::Types T>
  static inline toft::sliding_window_func<T> sliding_window_ptr =
      mml_sliding_window<T>;

  // Pointer to the gemm std::function. Different defaults depending on compiler
  // flags
#if defined(USE_AVX_GEMM)
  template <TensorConcept::Types T>
  static inline toft::gemm_func<T> gemm_ptr = mml_gemm_avx<T>;
#elif defined(USE_AVX512_GEMM)
  template <TensorConcept::Types T>
  static inline toft::gemm_func<T> gemm_ptr = mml_gemm_avx512<T>;
#elif defined(USE_BLOCKED_GEMM)
  template <TensorConcept::Types T>
  static inline toft::gemm_func<T> gemm_ptr = mml_gemm_blocked<T>;
#else
  template <TensorConcept::Types T>
  static inline toft::gemm_func<T> gemm_ptr = mml_gemm_inner_product<T>;
#endif

  // Pointer to the gemm_onnx std::function.
  template <TensorConcept::Types T>
  static inline toft::gemm_onnx_func<T> gemm_onnx_ptr =
      mml_onnx_gemm_inner_product<T>;

  // Pointer to the add std::function.
  template <TensorConcept::Types T>
  static inline toft::add_func<T> add_ptr = mml_add<T>;

  // Pointer to the subtract std::function.
  template <TensorConcept::Types T>
  static inline toft::subtract_func<T> subtract_ptr = mml_subtract<T>;

  // Pointer to the multiply std::function.
  template <TensorConcept::Types T>
  static inline toft::multiply_func<T> multiply_ptr = mml_multiply<T>;

  // Pointer to the equals std::function.
  template <TensorConcept::Types T>
  static inline toft::equals_func<T> equals_ptr = mml_equals<T>;

  // Pointer to the elementwise std::function.
  template <TensorConcept::Types T>
  static inline toft::elementwise_func elementwise_ptr = mml_elementwise<T>;

  // Pointer to the elementwise_in_place std::function.
  template <TensorConcept::Types T>
  static inline toft::elementwise_in_place_func elementwise_in_place_ptr =
      mml_elementwise_in_place<T>;

  // Pointer to the arg_max std::function.
  template <TensorConcept::Types T>
  static inline toft::arg_max_func arg_max_ptr = mml_arg_max<T>;
};

#include "../datastructures/tensor_operations_module.tpp"