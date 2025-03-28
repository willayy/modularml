#pragma once

#include "nodes/sa_node.hpp"

/**
 * @class LogSoftMaxNode
 * @brief A class representing a LogSoftMax node in a computational graph.
 *
 * This class inherits from the Node class and represents the LogSoftMax node
 * in a computational graph. It performs the forward pass computation applying
 * SoftMax and Logarithm along the specified axis.
 */
class LogSoftMaxNode : public Node {
public:
  using T = std::variant<float, double>;

  /**
   * @brief Constructor for LogSoftMaxNode.
   *
   * @param X Shared pointer to the tensor X.
   * @param Y Shared pointer to the output tensor.
   * @param axis Integer representing along which axis LogSoftMax is applied to.
   * (default -1)
   */
  LogSoftMaxNode(std::string X,
    std::string Y, uli axis = -1);
  

  /**
   * @brief Constructor for LogSoftMaxNode from JSON.
   * 
   * @param node JSON object representing the LogSoftMax node.
   */
  LogSoftMaxNode(const json &node);

  /**
   * @brief Perform the forward pass computation using LogSoftMax activation
   * function.
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
  std::string X; // Input tensor X.
  std::string Y;       // Output tensor Y.
  uli axis;
};
