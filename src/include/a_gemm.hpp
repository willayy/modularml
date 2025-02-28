#pragma once

#include "globals.hpp"
#include "tensor.hpp"

#define ASSERT_ALLOWED_TYPES_GM(T) static_assert(std::is_arithmetic_v<T>, "Data structure type must be an arithmetic type.")

/// @brief Abstract class for classes that contain standard GEMM functions.
/// @tparam T the type of the data that the GEMM functions will operate on.
template <typename T>
class GemmModule {
 public:
  /// @brief Default constructor for GEMM class.
  GemmModule() = default;

  /// @brief Copy constructor for GEMM class.
  GemmModule(const GemmModule& other) = default;

  /// @brief Move constructor for GEMM class.
  GemmModule(GemmModule&& other) noexcept = default;

  /// @brief Abstract destructor for GEMM class.
  virtual ~GemmModule() = default;

  /**
   * @brief Basic CPU implementation of GEMM, with the inner product approach.
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_inner_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                  shared_ptr<Tensor<T>> A, int lda,
                                  shared_ptr<Tensor<T>> B, int ldb,
                                  T BETA,
                                  shared_ptr<Tensor<T>> C, int ldc) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the outer product approach.
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_outer_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                  shared_ptr<Tensor<T>> A, int lda,
                                  shared_ptr<Tensor<T>> B, int ldb,
                                  T BETA,
                                  shared_ptr<Tensor<T>> C, int ldc) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the row-wise product approach.
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_row_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                     shared_ptr<Tensor<T>> A, int lda,
                                     shared_ptr<Tensor<T>> B, int ldb,
                                     T BETA,
                                     shared_ptr<Tensor<T>> C, int ldc) = 0;

  /**
   * @brief Basic CPU implementation of GEMM, with the col-wise product approach.
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_col_wise_product(int TA, int TB, int M, int N, int K, T ALPHA,
                                     shared_ptr<Tensor<T>> A, int lda,
                                     shared_ptr<Tensor<T>> B, int ldb,
                                     T BETA,
                                     shared_ptr<Tensor<T>> C, int ldc) = 0;

  /**
   * @brief Blocked CPU implementation of GEMM.
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_blocked(int TA, int TB, int M, int N, int K, T ALPHA,
                            shared_ptr<Tensor<T>> A, int lda,
                            shared_ptr<Tensor<T>> B, int ldb,
                            T BETA,
                            shared_ptr<Tensor<T>> C, int ldc) = 0;

  /**
   * @brief Vectorized implementation of GEMM using SIMD thanks to AVX
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_avx(int TA, int TB, int M, int N, int K, T ALPHA,
                        shared_ptr<Tensor<T>> A, int lda,
                        shared_ptr<Tensor<T>> B, int ldb,
                        T BETA,
                        shared_ptr<Tensor<T>> C, int ldc) = 0;

  /**
   * @brief Vectorized implementation of GEMM using SIMD thanks to AVX512
   * Performs operation C := alpha*op( A )*op( B ) + beta*C
   * More detailed info in
   * https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_avx512(int TA, int TB, int M, int N, int K, T ALPHA,
                           shared_ptr<Tensor<T>> A, int lda,
                           shared_ptr<Tensor<T>> B, int ldb,
                           T BETA,
                           shared_ptr<Tensor<T>> C, int ldc) = 0;

  /**
   * @brief GEMM using Intel's Math Kernel Library
   * Performs operation C := alpha*op( A )*op( B ) + beta*C.
   * More detailed info in
   * https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html#cblas-gemm
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
   * @param C 1D array containing the result matrix (can be initialized to non-zero for addition).
   * @param ldc Specifies the first dimension of matrix C.
   */
  virtual void gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                              shared_ptr<Tensor<T>> A, int lda,
                              shared_ptr<Tensor<T>> B, int ldb,
                              T BETA,
                              shared_ptr<Tensor<T>> C, int ldc) = 0;

  virtual shared_ptr<GemmModule<float>> clone() const = 0;
};