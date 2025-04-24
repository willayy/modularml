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
 * @brief A class representing a MatMul node in a computational graph.
 *
 * This class inherits from the Node class and represents a Matrix
 * Multiplication node in a computational graph. It performs the forward pass
 * computation using GEMM inner product.
 *
 * @author Tim Carlsson (timca@chalmers.se)
 */
class MatMulNode : public Node {
 public:
  using T = std::variant<double, float, int32_t, int64_t, uint32_t, uint64_t>;

  /**
   * @brief Constructor for MatMul node.
   *
   * @param A Shared pointer to the tensor A.
   * @param B Shared pointer to the tensor B.
   * @param Y Shared pointer to the output tensor.
   */
  MatMulNode(const std::string &A, const std::string &B, const std::string &Y);

  /**
   * @brief Constructor for MatMul from JSON.
   *
   * @param node JSON object representing the MatMul node.
   */
  MatMulNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation of GEMM.
   *
   * This std::function performs the forward pass computation using the General
   * Matrix Multiply (GEMM) inner product.
   */
  void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

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
  std::string A;  // Input tensor A.
  std::string B;  // Input tensor B.

  // Output
  std::string Y;  // Output tensor.
};
