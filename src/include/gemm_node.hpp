#pragma once

#include "a_node.hpp"
#include "globals.hpp"
#include "mml_gemm.hpp"
#include "mml_tensor.hpp"

/**
 * @class GemmNode
 * @brief A class representing a GEMM node in a computational graph.
 *
 * This class inherits from the Node class and represents a General Matrix Multiply (GEMM) node
 * in a computational graph. It performs the forward pass computation using the GEMM inner product.
 */
template <typename T>
class GemmNode : public Node {
  static_assert(
      std::is_same_v<T, float> ||
          std::is_same_v<T, double> ||
          std::is_same_v<T, int32_t> ||
          std::is_same_v<T, int64_t> ||
          std::is_same_v<T, uint32_t> ||
          std::is_same_v<T, uint64_t>,
      "GemmNode_T supports only float, double, int32_t, int64_t, uint32_t, or uint64_t");

 public:
  using AbstractTensor = Tensor<T>;

  /**
   * @brief Constructor for GemmNode.
   *
   * @param A Shared pointer to the tensor A.
   * @param B Shared pointer to the tensor B.
   * @param Y Shared pointer to the output tensor.
   * @param C Optional shared pointer to the tensor C.
   * @param alpha Scalar multiplier for A * B.
   * @param beta Scalar multiplier for C.
   * @param transA Whether to transpose A (0 means false).
   * @param transB Whether to transpose B (0 means false).
   */
  GemmNode(shared_ptr<AbstractTensor> A,
           shared_ptr<AbstractTensor> B,
           shared_ptr<AbstractTensor> Y,
           optional<shared_ptr<AbstractTensor>> C = std::nullopt,
           float alpha = 1.0f,
           float beta = 1.0f,
           int transA = 0,
           int transB = 0);

  /**
   * @brief Perform the forward pass computation using GEMM inner product.
   *
   * This function performs the forward pass computation using the General Matrix Multiply (GEMM) inner product.
   */
  void forward() override;

  /**
   * @brief Check if the input(s) are filled.
   *
   * @return True if the input(s) are filled, false otherwise.
   */
  bool areInputsFilled() const override;

  /**
   * @brief Set the input(s) for the node.
   *
   * @param inputs The input data to be set, where A is inputs[0], B is inputs[1] and optionally C is inputs[2].
   */
  void setInputs(const array_mml<GeneralDataTypes>& inputs) override;

  /**
   * @brief Check if the output(s) are filled.
   *
   * @return True if the output(s) are filled, false otherwise.
   */
  bool areOutputsFilled() const override;

  /**
   * @brief Get the output of the node.
   *
   * @return The output data.
   */
  array_mml<GeneralDataTypes> getOutputs() const override;

 private:
  // Inputs
  shared_ptr<AbstractTensor> A;            // Input tensor A.
  shared_ptr<AbstractTensor> B;            // Input tensor B.
  optional<shared_ptr<AbstractTensor>> C;  // Optional tensor C.

  // Output
  shared_ptr<AbstractTensor> Y;  // Output tensor.

  // Attributes
  float alpha;  // Scalar multiplier for A * B.
  float beta;   // Scalar multiplier for C.
  int transA;   // Whether to transpose A (0: no, non-zero: yes).
  int transB;   // Whether to transpose B (0: no, non-zero: yes).
};

#include "../gemm_node.tpp"