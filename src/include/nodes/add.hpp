#pragma once

#include "nodes/a_node.hpp"

class AddNode : public Node {
public:
  using T = std::variant<double, float, int32_t, int64_t>;
  using TensorT =
      TensorVariant<T>; // Gets std::variant<std::shared_ptr<tensor<T>>,
                        // ...> from T

  /**
   * @brief Constructor for AddNode.
   *
   * @param A String hash ID to the first input tensor.
   * @param B String hash ID to the second input tensor.
   * @param C String hash ID to the output tensor.
   */
  AddNode(std::string A, std::string B, std::string C);

  /**
   * @brief Constructor for AddNode from JSON.
   *
   * @param node JSON object representing the Add node.
   */
  AddNode(const nlohmann::json &node);

  /**
   * @brief Performs element-wise binary addition in the two input tensors and
   * stores the result in the output tensor.
   */
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
  // tensors
  std::string A; // Input tensor A
  std::string B; // Input tensor B
  std::string C; // Output tensor C

  /**
   * @brief Helper std::function used when broadcasting addition is required.
   * Likely only temporary to be replaced with something that can be used in
   * multiple nodes instead.
   *
   * @return The output data.
   */
  void broadcast_addition(const TensorT &a_ptr, const TensorT &b_ptr,
                          const TensorT &c_ptr) const;
};
