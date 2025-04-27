#pragma once

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "datastructures/mml_array.hpp"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

/**
 * @class MatMulNode
 * @brief A node that performs matrix multiplication in a computational graph.
 *
 * The MatMulNode represents the mathematical operation of matrix multiplication
 * between two tensors. For 2D tensors, it performs the standard matrix product.
 * For higher-dimensional tensors, it applies batch matrix multiplication
 * according to broadcasting rules.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
class MatMulNode : public Node {
 public:
  /**
   * @brief Type alias for supported numeric types in matrix multiplication
   */
  using T = std::variant<double, float, int32_t, int64_t, uint32_t, uint64_t>;

  /**
   * @brief Constructor for MatMulNode with explicit tensor names.
   *
   * @param A Name of the first input tensor
   * @param B Name of the second input tensor
   * @param Y Name of the output tensor that will store the result
   */
  MatMulNode(const std::string &A, const std::string &B, const std::string &Y);

  /**
   * @brief Constructor for MatMulNode from JSON representation.
   *
   * This constructor parses the JSON definition from an ONNX or similar model
   * format to extract the tensor names for matrix multiplication.
   *
   * @param node JSON object representing the MatMul node definition
   */
  MatMulNode(const nlohmann::json &node);

  /**
   * @brief Performs the forward pass computation of matrix multiplication.
   *
   * This method retrieves the input tensors from the iomap, performs matrix
   * multiplication using the General Matrix Multiply (GEMM) implementation,
   * and stores the result in the output tensor.
   *
   * The operation follows standard matrix multiplication rules:
   * - For 2D tensors: C = A * B where A has shape (M, K) and B has shape (K, N)
   * - For higher dimensions: batch multiplication with broadcasting
   *
   * @param iomap Map containing input and output tensors indexed by name
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

  /**
   * @brief Gets the names of input tensors required by this node.
   *
   * @return A vector containing the names of the two input tensors (A and B)
   */
  std::vector<std::string> getInputs() override;

  /**
   * @brief Gets the name of the output tensor produced by this node.
   *
   * @return A vector containing the single output tensor name (Y)
   */
  std::vector<std::string> getOutputs() override;

 private:
  /**
   * @brief Name of the first input tensor A
   *
   * For 2D tensors, A should have shape (M, K)
   */
  std::string A;

  /**
   * @brief Name of the second input tensor B
   *
   * For 2D tensors, B should have shape (K, N)
   */
  std::string B;

  /**
   * @brief Name of the output tensor Y
   *
   * For 2D tensors, Y will have shape (M, N)
   */
  std::string Y;
};
