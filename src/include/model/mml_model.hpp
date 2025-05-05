#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "model/a_model.hpp"
#include "nodes/a_node.hpp"

/**
 * @class Model_mml
 * @brief A concrete implementation of a modular machine learning model.
 *
 * This class inherits from the Model base class and represents a machine
 * learning model with a directed graph of computational nodes. It provides
 * functionality to add nodes to the graph and run inference through the graph
 * in topological order.
 */
class Model_mml : public Model {
 public:
  /**
   * @brief Default constructor for Model_mml.
   *
   * Initializes an empty model with no nodes, inputs, or outputs.
   */
  Model_mml() = default;

  /**
   * @brief Constructor for Model_mml with initial nodes and I/O specification.
   *
   * @param initialNodes A vector of shared pointers to Node objects that form
   * the computational graph
   * @param iomap A map of tensor names to their corresponding tensor objects
   * for storing intermediate values
   * @param inputs A vector of input tensor names that the model expects
   * @param outputs A vector of output tensor names that the model produces
   */
  explicit Model_mml(std::vector<std::shared_ptr<Node>> initialNodes,
                     std::unordered_map<std::string, GeneralDataTypes> iomap,
                     std::vector<std::string> inputs,
                     std::vector<std::string> outputs)
      : nodes(std::move(initialNodes)),
        iomap(std::move(iomap)),
        inputs(std::move(inputs)),
        outputs(std::move(outputs)) {}

  /**
   * @brief Adds a node to the model's computational graph.
   *
   * This method appends a new computational node to the end of the nodes
   * vector. Note: This does not guarantee correct ordering for inference; the
   * topological sort will determine the execution order.
   *
   * @param node A shared pointer to a Node object to be added to the graph
   */
  void addNode(std::shared_ptr<Node> node) { nodes.push_back(std::move(node)); }

  /**
   * @brief Runs inference on the model's computational graph.
   *
   * This method performs the following steps:
   * 1. Updates the internal iomap with the provided input tensors
   * 2. Performs a topological sort of the nodes to determine execution order
   * 3. Executes each node's forward method in the determined order
   * 4. Returns the output tensors as specified in the model's outputs list
   *
   * @param inputs A map of input tensor names to their corresponding tensor
   * values
   * @return A map of output tensor names to their computed tensor values
   */
  std::unordered_map<std::string, GeneralDataTypes> infer(
      const std::unordered_map<std::string, GeneralDataTypes> &inputs) override;

 private:
  /**
   * @brief The computational nodes that form the model graph
   */
  std::vector<std::shared_ptr<Node>> nodes;

  /**
   * @brief Map of tensor names to tensor objects for storing intermediate
   * values
   */
  std::unordered_map<std::string, GeneralDataTypes> iomap;

  /**
   * @brief Names of input tensors that the model expects
   */
  std::vector<std::string> inputs;

  /**
   * @brief Names of output tensors that the model produces
   */
  std::vector<std::string> outputs;

  /**
   * @brief Performs a topological sort of the model's nodes
   *
   * This helper method sorts the nodes into levels where nodes in the same
   * level can be executed in parallel, and nodes in later levels depend on
   * nodes in earlier levels.
   *
   * @return A vector of vectors, where each inner vector contains nodes that
   * can be executed in parallel
   */
  std::vector<std::vector<std::shared_ptr<Node>>> topologicalSort();
};