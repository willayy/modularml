#pragma once

#include "a_tensor.hpp"
#include "globals.hpp"

template <typename T>
class OnnxGemm_mml : public OnnxGemmModule<T> {
 public:
    OnnxGemmModule() = default;

    OnnxGemmModule(const OnnxGemmModule& other) = default;

    OnnxGemmModule(OnnxGemmModule&& other) noexcept = default;

    virtual ~OnnxGemmModule() = default;

    shared_ptr<Tensor<T>> gemm_inner_product(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                    shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                    optional<shared_ptr<Tensor<T>>> C = nullopt) override;

    shared_ptr<Tensor<T>> gemm_outer_product(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                        shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                        optional<shared_ptr<Tensor<T>>> C = nullopt) override;

    shared_ptr<Tensor<T>> gemm_row_wise_product(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                        shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                        optional<shared_ptr<Tensor<T>>> C = nullopt) override;

    shared_ptr<Tensor<T>> gemm_col_wise_product(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                        shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                        optional<shared_ptr<Tensor<T>>> C = nullopt) override;

    shared_ptr<Tensor<T>> gemm_blocked(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                optional<shared_ptr<Tensor<T>>> C = nullopt) override;

    shared_ptr<Tensor<T>> gemm_avx(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                optional<shared_ptr<Tensor<T>>> C = nullopt) override;

    shared_ptr<Tensor<T>> gemm_avx512(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                optional<shared_ptr<Tensor<T>>> C = nullopt) override;

    shared_ptr<Tensor<T>> gemm_intel_MKL(float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0,
                                                shared_ptr<Tensor<T>> A, shared_ptr<Tensor<T>> B,
                                                optional<shared_ptr<Tensor<T>>> C = nullopt) override;

};

// Include the implementation of the templated class
#include "../mml_onnx_gemm.tpp"