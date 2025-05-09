#include "datastructures/tensor_operations.hpp"

template <typename T>
void TensorOperations<T>::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                               T BETA, std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  int block_size = 64;  // This depends on the CPU architecture - We can look
                        // into having the size of this be dynamically fetched
  if (!TA && !TB) {
    int i;
    int j;
    int jj;
    int k;
    int kk;
    int i_col;
    int k_col;
    int i_col_out;

    for (jj = 0; jj < N; jj += block_size) {
      for (kk = 0; kk < K; kk += block_size) {
        for (i = 0; i < M; i++) {
          i_col = i * lda;
          i_col_out = i * ldc;
          for (int j = jj; j < std::min(jj + block_size, N); j++) {
            T acc = BETA * (*C)[i_col_out + j];
            for (int k = kk; k < std::min(kk + block_size, K); k++) {
              k_col = k * ldb;
              acc += ALPHA * (*A)[i_col + k] * (*B)[k_col + j];
            }
            (*C)[i_col_out + j] = acc;
          }
        }
      }
    }
  } else if (TA && !TB) {
    throw std::invalid_argument(
        "Transposition not yet supported in GEMM blocked.");
  } else if (!TA && TB) {
    throw std::invalid_argument(
        "Transposition not yet supported in GEMM blocked.");
  } else {
    throw std::invalid_argument(
        "Transposition not yet supported in GEMM blocked.");
  }
  return;
}

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE