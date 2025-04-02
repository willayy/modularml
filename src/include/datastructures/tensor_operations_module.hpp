#pragma once
#include "a_tensor.hpp"
#include "globals.hpp"
#include "tensor_operation_functions.hpp"

#define ASSERT_ALLOWED_TYPES_TOM(T)                                             \
  static_assert(std::is_arithmetic_v<T>,                                       \
                "TensorOperationModule type must be arithmetic.")
/**
 *   A module for performing arithmetic operations on tensor structures. Your
 * modular toolbox for all operations on tensors.
 *    @param T the data type (arithmetic).
 */
class TensorOperationsModule {

public:
  TensorOperationsModule(const TensorOperationsModule &) = delete;
  TensorOperationsModule &operator=(const TensorOperationsModule &) = delete;

  /**
   * @brief General matrix multiplication (GEMM) function.
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
  template <typename T>
  static void gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                   shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B,
                   int ldb, T BETA, shared_ptr<Tensor<T>> C, int ldc);

  /**
   * @brief General matrix multiplication (GEMM) function using the ONNX
   * standard. Performs operation Y := alpha * A * B + beta * C (optional) More
   * detailed info in https://onnx.ai/onnx/operators/onnx__Gemm.html
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
   * @return Output tensor Y. Output tensor of shape (M, N). */
  template <typename T>
  static shared_ptr<Tensor<T>>
  gemm_onnx(shared_ptr<Tensor<T>> A = nullptr,
            shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
            float beta = 1.0, int transA = 0, int transB = 0,
            optional<shared_ptr<Tensor<T>>> C = nullopt);

  /**
   * @brief Adds two tensors element-wise.
   * Performs operation c = a + b.
   * @param a First tensor.
   * @param b Second tensor.
   * @param c Output tensor. */
  template <typename T>
  static void add(const shared_ptr<const Tensor<T>> a,
                  const shared_ptr<const Tensor<T>> b, shared_ptr<Tensor<T>> c);

  /**
   * @brief Subtracts two tensors element-wise.
   * Performs operation c = a - b.
   * @param a First tensor.
   * @param b Second tensor.
   * @param c Output tensor. */
  template <typename T>
  static void subtract(const shared_ptr<Tensor<T>> a,
                       const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c);

  /**
   * @brief Multiplies a tensor by a scalar.
   * Performs operation c = a * b.
   * @param a Input tensor.
   * @param b Scalar.
   * @param c Output tensor. */
  template <typename T>
  static void multiply(const shared_ptr<Tensor<T>> a, const T b,
                       shared_ptr<Tensor<T>> c);

  /**
   * @brief Compares two tensors element-wise.
   * @param a First tensor.
   * @param b Second tensor.
   * @return True if the tensors are equal, false otherwise. */
  template <typename T>
  static bool equals(const shared_ptr<Tensor<T>> a,
                     const shared_ptr<Tensor<T>> b);

  /**
   * @brief Applies a function element-wise to a tensor.
   * @param a Input tensor.
   * @param f Function to apply.
   * @param c Output tensor. */
  template <typename T>
  static void elementwise(const shared_ptr<const Tensor<T>> a,
                          const function<T(T)> f,
                          const shared_ptr<Tensor<T>> c);

  /**
   * @brief Applies a function element-wise to a tensor in place.
   * @param a Input tensor.
   * @param f Function to apply. */
  template <typename T>
  static void elementwise_in_place(const shared_ptr<Tensor<T>> a,
                                   const function<T(T)> f);

  /**
   * @brief Gets the maximum value of a tensor along a given axis.
   * @param a Input tensor.
   * @return The maximum value along the specified axis.
   */
  template <typename T> static int arg_max(const shared_ptr<const Tensor<T>> a);

  /**
   * @brief Sets the gemm function pointer.
   * @param ptr Function pointer to the gemm implementation.
   */
  template <typename T>
  static void set_gemm_ptr(void (*ptr)(int TA, int TB, int M, int N, int K,
                                       T ALPHA, shared_ptr<Tensor<T>> A,
                                       int lda, shared_ptr<Tensor<T>> B,
                                       int ldb, T BETA, shared_ptr<Tensor<T>> C,
                                       int ldc));

  /**
   * @brief Sets the gemm_onnx function pointer.
   * @param ptr Function pointer to the gemm_onnx implementation.
   */
  template <typename T>
  static void set_gemm_onnx_ptr(shared_ptr<Tensor<T>> (*ptr)(
      shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha, float beta,
      int transA, int transB, optional<shared_ptr<Tensor<T>>> C));

  /**
   * @brief Sets the add function pointer.
   * @param ptr Function pointer to the add implementation.
   */
  template <typename T>
  static void set_add_ptr(void (*ptr)(const shared_ptr<const Tensor<T>> a,
                                      const shared_ptr<const Tensor<T>> b,
                                      shared_ptr<Tensor<T>> c));

  /**
   * @brief Sets the subtract function pointer.
   * @param ptr Function pointer to the subtract implementation.
   */
  template <typename T>
  static void set_subtract_ptr(void (*ptr)(const shared_ptr<Tensor<T>> a,
                                           const shared_ptr<Tensor<T>> b,
                                           shared_ptr<Tensor<T>> c));

  /**
   * @brief Sets the multiply function pointer.
   * @param ptr Function pointer to the multiply implementation.
   */
  template <typename T>
  static void set_multiply_ptr(void (*ptr)(const shared_ptr<Tensor<T>> a,
                                           const T b, shared_ptr<Tensor<T>> c));

  /**
   * @brief Sets the equals function pointer.
   * @param ptr Function pointer to the equals implementation.
   */
  template <typename T>
  static void set_equals_ptr(bool (*ptr)(const shared_ptr<Tensor<T>> a,
                                         const shared_ptr<Tensor<T>> b));

  /**
   * @brief Sets the elementwise function pointer.
   * @param ptr Function pointer to the elementwise implementation.
   */
  template <typename T>
  static void
  set_elementwise_ptr(void (*ptr)(const shared_ptr<const Tensor<T>> a,
                                  const function<T(T)> &f,
                                  const shared_ptr<Tensor<T>> c));

  /**
   * @brief Sets the elementwise_in_place function pointer.
   * @param ptr Function pointer to the elementwise_in_place implementation.
   */
  template <typename T>
  static void set_elementwise_in_place_ptr(
      void (*ptr)(const shared_ptr<Tensor<T>> a, const function<T(T)> &f));

  /**
   * @brief Sets the arg_max function pointer.
   * @param ptr Function pointer to the arg_max implementation.
   */
  template <typename T>
  static void set_arg_max_ptr(int (*ptr)(const shared_ptr<const Tensor<T>> a));

private:
  // Private constructor.
  TensorOperationsModule() {}

  // Pointer to the gemm function.
  template <typename T>
  static void (*gemm_ptr)(int TA, int TB, int M, int N, int K, T ALPHA,
                          shared_ptr<Tensor<T>> A, int lda,
                          shared_ptr<Tensor<T>> B, int ldb, T BETA,
                          shared_ptr<Tensor<T>> C,
                          int ldc) = mml_gemm_inner_product;

  // Pointer to the gemm_onnx function.
  template <typename T>
  static shared_ptr<Tensor<T>> (*gemm_onnx_ptr)(
      shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B, float alpha, float beta,
      int transA, int transB,
      optional<shared_ptr<Tensor<T>>> C) = mml_onnx_gemm_inner_product;

  // Pointer to the add function.
  template <typename T>
  static void (*add_ptr)(const shared_ptr<const Tensor<T>> a,
                         const shared_ptr<const Tensor<T>> b,
                         shared_ptr<Tensor<T>> c) = mml_add;

  // Pointer to the subtract function.
  template <typename T>
  static void (*subtract_ptr)(const shared_ptr<Tensor<T>> a,
                              const shared_ptr<Tensor<T>> b,
                              shared_ptr<Tensor<T>> c) = mml_subtract;

  // Pointer to the multiply function.
  template <typename T>
  static void (*multiply_ptr)(const shared_ptr<Tensor<T>> a, const T b,
                              shared_ptr<Tensor<T>> c) = mml_multiply;

  // Pointer to the equals function.
  template <typename T>
  static bool (*equals_ptr)(const shared_ptr<Tensor<T>> a,
                            const shared_ptr<Tensor<T>> b) = mml_equals;

  // Pointer to the elementwise function.
  template <typename T>
  static void (*elementwise_ptr)(
      const shared_ptr<const Tensor<T>> a, const function<T(T)> &f,
      const shared_ptr<Tensor<T>> c) = mml_elementwise;

  // Pointer to the elementwise_in_place function.
  template <typename T>
  static void (*elementwise_in_place_ptr)(const shared_ptr<Tensor<T>> a,
                                          const function<T(T)> &f) =
      mml_elementwise_in_place;

  // Pointer to the arg_max function.
  template <typename T>
  static int (*arg_max_ptr)(const shared_ptr<const Tensor<T>> a) = mml_arg_max;
};