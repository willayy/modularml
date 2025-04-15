#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>  // IWYU pragma: keep

#include "backend/mml_gemm.hpp"
#include "datastructures/mml_tensor.hpp"

template <typename T>
class OnnxGemm_mml : public OnnxGemmModule<T> {
 public:
  [[deprecated("Use TensorOperationsModule instead")]]
  OnnxGemm_mml() = default;
  [[deprecated("Use TensorOperationsModule instead")]]
  OnnxGemm_mml(const OnnxGemm_mml &other) = default;
  [[deprecated("Use TensorOperationsModule instead")]]
  OnnxGemm_mml(OnnxGemm_mml &&other) noexcept = default;
  [[deprecated("Use TensorOperationsModule instead")]]
  virtual ~OnnxGemm_mml() = default;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_inner_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_outer_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_row_wise_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_col_wise_product(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_blocked(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_avx(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_avx512(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
  [[deprecated("Use TensorOperationsModule instead")]]
  std::shared_ptr<Tensor<T>> gemm_intel_MKL(
      std::shared_ptr<Tensor<T>> A = nullptr,
      std::shared_ptr<Tensor<T>> B = nullptr, float alpha = 1.0,
      float beta = 1.0, int transA = 0, int transB = 0,
      std::optional<std::shared_ptr<Tensor<T>>> C = std::nullopt) override;
};

#include "../backend/mml_onnx_gemm.tpp"