#include "datastructures/tensor_operations.hpp"

template <typename T>
void TensorOperations<T>::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                               T BETA, std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  int k_col;
  int i_col_out;

  if (TA == 1) A = A->transpose();
  if (TB == 1) B = B->transpose();

  for (int i = 0; i < M; i++) {
    i_col_out = i * ldc;

    for (int j = 0; j < N; j++) {
      (*C)[i_col_out + j] = ((T)BETA) * (*C)[i_col_out + j];
    }
    for (int k = 0; k < K; k++) {
      k_col = k * ldb;

      for (int j = 0; j < N; j++) {
        (*C)[i_col_out + j] += ((T)ALPHA) * (*A)[i * lda + k] * (*B)[k_col + j];
      }
    }
  }

  return;
}

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE