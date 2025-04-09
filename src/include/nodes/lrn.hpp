#pragma once

#include "nodes/a_node.hpp"

/**
 * @class LRNNode_mml
 * @brief Performs Local Response Normalization
 * @details LRNNode_mml performs Local Response Normalization according to the
 * ONNX specifications. It normalizes the tensor across local input regions. The
 * local region is defined across the channels.
 * @tparam The datatype i the tensor. Accepts float and double.
 */
class LRNNode_mml : public Node {
public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for LRNNode_mml
   * @param input A shared pointer to the input tensor.
   * @param size (Required) The number of channels to sum over. Must be at
   * least 1.
   * @param alpha (default = 0.0001) Scaling parameter
   * @param beta (default = 0.75) The exponent. Must be at least 0.
   * @param bias (default = 1.0) Bias to avoid division with 0. Must be at least
   * 0.001.
   *
   */
  LRNNode_mml(std::string X, std::string Y, size_t size, float alpha = 0.0001f,
              float beta = 0.75f, float bias = 1.0f);

  LRNNode_mml(const nlohmann::json &node);

  void
  forward(std::unordered_map<std::string, GeneralDataTypes> &iomap) override;

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
  ///@brief Shared pointer to the input tensor
  std::string X;

  ///@brief Shared pointer to the output tensor
  std::string Y;

  ///@brief Scaling parameter
  float alpha;

  ///@brief The exponent
  float beta;

  ///@brief To avoid division by zero
  float bias;

  ///@brief Number of channels to sum over
  size_t size;
};
