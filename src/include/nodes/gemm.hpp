#pragma once

#include "nodes/a_node.hpp"

/**
 * @class GemmNode
 * @brief A class representing a GEMM node in a computational graph.
 *
 * This class inherits from the Node class and represents a General Matrix Multiply (GEMM) node
 * in a computational graph. It performs the forward pass computation using the GEMM inner product.
 */
class GemmNode : public Node {
 public:
  using T = std::variant<double, float, int32_t, int64_t, uint32_t, uint64_t>;

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
  GemmNode(std::string A,
           std::string B,
           std::string Y,
           optional<std::string> C = std::nullopt,
           float alpha = 1.0f,
           float beta = 1.0f,
           int transA = 0,
           int transB = 0);
  
  /**
   * @brief Constructor for GemmNode from JSON.
   * 
   * @param node JSON object representing the Gemm node.
   */
  GemmNode(const json& node);

  /**
   * @brief Perform the forward pass computation using GEMM inner product.
   *
   * This function performs the forward pass computation using the General Matrix Multiply (GEMM) inner product.
   */
  void forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) override;

  /**
   * @brief Get inputs.
   * 
   * @return The names of the inputs to the node.
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Get outputs.
   * 
   * @return The names of the outputs to the node.
   */
  std::vector<std::string> getOutputs() override;


 private:
  // Inputs
  std::string A;            // Input tensor A.
  std::string B;            // Input tensor B.
  optional<std::string> C;  // Optional tensor C.

  // Output
  std::string Y;  // Output tensor.

  // Attributes
  float alpha;  // Scalar multiplier for A * B.
  float beta;   // Scalar multiplier for C.
  int transA;   // Whether to transpose A (0: no, non-zero: yes).
  int transB;   // Whether to transpose B (0: no, non-zero: yes).
};
