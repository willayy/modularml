#pragma once

#include "globals.hpp"
#include "backend/mml_gemm.hpp"
#include "datastructures/mml_tensor.hpp"

template <typename T>
class OnnxGemm_mml : public OnnxGemmModule<T> {
 public:
  OnnxGemm_mml() = default;

  OnnxGemm_mml(const OnnxGemm_mml& other) = default;

  OnnxGemm_mml(OnnxGemm_mml&& other) noexcept = default;

  virtual ~OnnxGemm_mml() = default;

  shared_ptr<Tensor<T>> gemm_inner_product(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                           float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                           optional<shared_ptr<Tensor<T>>> C = nullopt) override;

  shared_ptr<Tensor<T>> gemm_outer_product(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                           float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                           optional<shared_ptr<Tensor<T>>> C = nullopt) override;

  shared_ptr<Tensor<T>> gemm_row_wise_product(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                              float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                              optional<shared_ptr<Tensor<T>>> C = nullopt) override;

  shared_ptr<Tensor<T>> gemm_col_wise_product(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                              float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                              optional<shared_ptr<Tensor<T>>> C = nullopt) override;

  shared_ptr<Tensor<T>> gemm_blocked(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                     float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                     optional<shared_ptr<Tensor<T>>> C = nullopt) override;

  shared_ptr<Tensor<T>> gemm_avx(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                 float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                 optional<shared_ptr<Tensor<T>>> C = nullopt) override;

  shared_ptr<Tensor<T>> gemm_avx512(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                    float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                    optional<shared_ptr<Tensor<T>>> C = nullopt) override;

  shared_ptr<Tensor<T>> gemm_intel_MKL(shared_ptr<Tensor<T>> A = nullptr, shared_ptr<Tensor<T>> B = nullptr,
                                       float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                       optional<shared_ptr<Tensor<T>>> C = nullopt) override;
};

#include "../backend/mml_onnx_gemm.tpp"