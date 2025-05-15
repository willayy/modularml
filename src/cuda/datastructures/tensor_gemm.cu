#include <cuda_runtime.h>

#include "datastructures/tensor_operations.hpp"

template <typename T>
__global__ void gemmKernel(int M, int N, int K, T ALPHA, T BETA, const T* A,
                           int lda, const T* B, int ldb, T* C, int ldc) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    T sum = 0;
    for (int i = 0; i < K; ++i) sum += A[row * lda + i] * B[i * ldb + col];
    C[row * ldc + col] = BETA * C[row * ldc + col] + ALPHA * sum;
  }
}

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); \
  } while (0)

template <typename T>
void TensorOperations<T>::gemm(int TA, int TB, int M, int N, int K, T ALPHA,
                               T BETA, std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  const auto A_shape = A->get_shape();
  const auto B_shape = B->get_shape();
  const auto C_shape = C->get_shape();

  if (C_shape[0] != M || C_shape[1] != N) {
    throw std::invalid_argument("Output matrix C dimensions don't match M×N");
  }

  if (!TA && !TB) {
    if (A_shape[0] != M || A_shape[1] != K || B_shape[0] != K ||
        B_shape[1] != N) {
      throw std::invalid_argument(
          "Input matrix dimensions don't match for multiplication");
    }
  } else if (TA && !TB) {
    if (A_shape[0] != K || A_shape[1] != M || B_shape[0] != K ||
        B_shape[1] != N) {
      throw std::invalid_argument(
          "Input matrix dimensions don't match for multiplication with A "
          "transposed");
    }
  } else if (!TA && TB) {
    if (A_shape[0] != M || A_shape[1] != K || B_shape[0] != N ||
        B_shape[1] != K) {
      throw std::invalid_argument(
          "Input matrix dimensions don't match for multiplication with B "
          "transposed");
    }
  } else {
    if (A_shape[0] != K || A_shape[1] != M || B_shape[0] != N ||
        B_shape[1] != K) {
      throw std::invalid_argument(
          "Input matrix dimensions don't match for multiplication with both "
          "matrices transposed");
    }
  }

  if (TA) A = A->transpose();
  if (TB) B = B->transpose();

  const T* hA = A->get_data().get();
  const T* hB = B->get_data().get();
  T* hC = const_cast<T*>(C->get_data().get());

  T *dA = nullptr, *dB = nullptr, *dC = nullptr;
  size_t sizeA = sizeof(T) * size_t(lda) * size_t(M);
  size_t sizeB = sizeof(T) * size_t(ldb) * size_t(K);
  size_t sizeC = sizeof(T) * size_t(ldc) * size_t(M);
  CUDA_CHECK(cudaMalloc(&dA, sizeA));
  CUDA_CHECK(cudaMalloc(&dB, sizeB));
  CUDA_CHECK(cudaMalloc(&dC, sizeC));

  CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC, sizeC, cudaMemcpyHostToDevice));

  constexpr int TILE = 16;
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemmKernel<T>
      <<<grid, block>>>(M, N, K, ALPHA, BETA, dA, lda, dB, ldb, dC, ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

#define TYPE(DT) _TENSOR_OPERATIONS(DT)
#include "types_integer.txt"
#include "types_real.txt"
#undef TYPE