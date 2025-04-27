#include "datastructures/tensor_concept.hpp"
#include "datastructures/tensor_intel_operation_functions.hpp"

#if defined(USE_INTEL_MKL_GEMM)
template <TensorConcept::Types T>
static void mml_gemm_intel_MKL(int TA, int TB, int M, int N, int K, T ALPHA,
                               std::shared_ptr<Tensor<T>> A, int lda,
                               std::shared_ptr<Tensor<T>> B, int ldb, T BETA,
                               std::shared_ptr<Tensor<T>> C, int ldc) {
  std::invalid_argument("Intel MKL GEMM not yet supported.");
}
#endif