#pragma once
#include "a_tensor.hpp"
#include "globals.hpp"
#include "tensor_operation_functions.hpp"

#define ASSERT_ALLOWED_TYPES_AR(T)                                             \
  static_assert(std::is_arithmetic_v<T>,                                       \
                "TensorOperationModule type must be arithmetic.")
/**
 *   A module for performing arithmetic operations on tensor structures. Your
 * modular toolbox for all operations on tensors.
 *    @param T the data type (arithmetic).
 */
template <typename T> class TensorOperationsModule {

public:
  /**
   * @brief Get the instance of the TensorOperationsModule.
   * @return The instance of the TensorOperationsModule. */
  static TensorOperationsModule &getInstance();
  // Delete copy constructor and assignment operator.
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
  void gemm(int TA, int TB, int M, int N, int K, T ALPHA,
            shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B, int ldb,
            T BETA, shared_ptr<Tensor<T>> C, int ldc) = 0;

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
  shared_ptr<Tensor<T>>
  gemm_onnx(shared_ptr<Tensor<T>> A = nullptr,
            shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
            float beta = 1.0, int transA = 0, int transB = 0,
            optional<shared_ptr<Tensor<T>>> C = nullopt) = 0;

  /**
   * @brief Adds two tensors element-wise.
   * Performs operation c = a + b.
   * @param a First tensor.
   * @param b Second tensor.
   * @param c Output tensor. */
  void add(const shared_ptr<const Tensor<T>> a,
           const shared_ptr<const Tensor<T>> b, shared_ptr<Tensor<T>> c) const;

  /**
   * @brief Subtracts two tensors element-wise.
   * Performs operation c = a - b.
   * @param a First tensor.
   * @param b Second tensor.
   * @param c Output tensor. */
  void subtract(const shared_ptr<Tensor<T>> a, const shared_ptr<Tensor<T>> b,
                shared_ptr<Tensor<T>> c) const;

  /**
   * @brief Multiplies a tensor by a scalar.
   * Performs operation c = a * b.
   * @param a Input tensor.
   * @param b Scalar.
   * @param c Output tensor. */
  void multiply(const shared_ptr<Tensor<T>> a, const T b,
                shared_ptr<Tensor<T>> c) const;

  /**
   * @brief Compares two tensors element-wise.
   * @param a First tensor.
   * @param b Second tensor.
   * @return True if the tensors are equal, false otherwise. */
  bool equals(const shared_ptr<Tensor<T>> a,
              const shared_ptr<Tensor<T>> b) const;

  /**
   * @brief Applies a function element-wise to a tensor.
   * @param a Input tensor.
   * @param f Function to apply.
   * @param c Output tensor. */
  void elementwise(const shared_ptr<const Tensor<T>> a, function<T(T)> f,
                   const shared_ptr<Tensor<T>> c) const;

  /**
   * @brief Applies a function element-wise to a tensor in place.
   * @param a Input tensor.
   * @param f Function to apply. */
  void elementwise_in_place(const shared_ptr<Tensor<T>> a,
                            function<T(T)> f) const;

  void set_operation_func(string id, function<void()> func);

private:
  // Private constructor.
  TensorOperationsModule() {
    this->gemm_ptr = mml_gemm_inner_product;
    this->gemm_onnx_ptr = mml_onnx_gemm_inner_product;
    this->add_ptr = mml_add;
    this->subtract_ptr = mml_subtract;
    this->multiply_ptr = mml_multiply;
    this->equals_ptr = mml_equals;
    this->elementwise_ptr = mml_elementwise;
    this->elementwise_in_place_ptr = mml_elementwise_in_place;
  }

  // Pointer to the gemm function.
  void (*gemm_ptr)(int TA, int TB, int M, int N, int K, T ALPHA,
                   shared_ptr<Tensor<T>> A, int lda, shared_ptr<Tensor<T>> B,
                   int ldb, T BETA, shared_ptr<Tensor<T>> C, int ldc);
  // Pointer to the gemm_onnx function.
  shared_ptr<Tensor<T>> (*gemm_onnx_ptr)(
      shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
      float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
      optional<shared_ptr<Tensor<T>>> C = nullopt);
  // Pointer to the add function.
  void (*add_ptr)(const shared_ptr<const Tensor<T>> a,
                  const shared_ptr<const Tensor<T>> b, shared_ptr<Tensor<T>> c);
  // Pointer to the subtract function.
  void (*subtract_ptr)(const shared_ptr<Tensor<T>> a,
                       const shared_ptr<Tensor<T>> b, shared_ptr<Tensor<T>> c);
  // Pointer to the multiply function.
  void (*multiply_ptr)(const shared_ptr<Tensor<T>> a, const T b,
                       shared_ptr<Tensor<T>> c);
  // Pointer to the equals function.
  bool (*equals_ptr)(const shared_ptr<Tensor<T>> a,
                     const shared_ptr<Tensor<T>> b);
  // Pointer to the elementwise function.
  void (*elementwise_ptr)(const shared_ptr<const Tensor<T>> a, function<T(T)> f,
                          const shared_ptr<Tensor<T>> c);
  // Pointer to the elementwise_in_place function.
  void (*elementwise_in_place_ptr)(const shared_ptr<Tensor<T>> a,
                                   function<T(T)> f);
};

#include "../tensor_operations_module.tpp"