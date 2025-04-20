#pragma once

#include <stdint.h>

#include <optional>
#include <string>
#include <variant>

#include "nlohmann/json_fwd.hpp"
#include "nodes/a_node.hpp"

/**
 * @class TransposeNode
 * @brief A class representing a Transpose node in a computational graph.
 *
 * This class inherits from the Node class and represents a transpose operation as defined in onnx
 * 
 * @author Tim Carlsson (timca@chalmers.se)
 */
class TransposeNode : public Node {
 public:
  using T = std::variant<double, float, int32_t, int64_t, uint32_t, uint64_t>;

  /**
   * @brief Constructor for GemmNode.
   *
   * @param A Shared pointer to the tensor A.
   * @param Y Shared pointer to the output tensor.
   * @param perm A list of integers defining how to permute the axi
   */
  TransposeNode(std::string A, std::string Y, const std::vector<int> perm);

  /**
   * @brief Constructor for TransposeNode from JSON.
   *
   * @param node JSON object representing the Gemm node.
   */
  TransposeNode(const nlohmann::json &node);

  /**
   * @brief Perform the forward pass computation of GEMM.
   *
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
  std::string A;   // Input tensor A.

  // Output
  std::string Y;  // Output tensor.

  // Attributes

  /**
   * @brief A list of integers. By default, reverse the dimensions, 
   * otherwise permute the axes according to the values given. 
   * Its length must be equal to the rank of the input.
   */
  std::vector<int> perm; // 
};