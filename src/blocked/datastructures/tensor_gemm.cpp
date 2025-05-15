#include "datastructures/tensor_operations.hpp"

template <typename T>
void TensorOperations<T>::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                               T BETA, std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  int block_size = 64;  // Can be tuned or made adaptive later

  if (!TA && !TB) {
    for (int ii = 0; ii < M; ii += block_size) {
      for (int kk = 0; kk < K; kk += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
          for (int i = ii; i < std::min(ii + block_size, M); i++) {
            int i_col_A = i * lda;
            int i_col_C = i * ldc;
            for (int j = jj; j < std::min(jj + block_size, N); j++) {
              T acc = (kk == 0 ? BETA * (*C)[i_col_C + j] : (*C)[i_col_C + j]);
              for (int k = kk; k < std::min(kk + block_size, K); k++) {
                // Access B row-wise now: B[k * ldb + j]
                acc += ALPHA * (*A)[i_col_A + k] * (*B)[k * ldb + j];
              }
              (*C)[i_col_C + j] = acc;
            }
          }
        }
      }
    }
  } else {
    throw std::invalid_argument("Transposition not supported in blocked GEMM.");
  }
}

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE