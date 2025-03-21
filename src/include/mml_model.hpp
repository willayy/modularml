#pragma once

#include "a_model.hpp"
#include "a_node.hpp"


/**
 * @class Model_mml
 * @brief A class representing a modular machine learning model.
 * 
 * This class inherits from the Model class and represents a machine learning model
 * with a graph of nodes. It provides functionality to add nodes to the graph and 
 * run inference on the graph.
 */
class Model_mml: public Model {
public:
    /**
     * @brief Default constructor for Model_mml.
     * 
     * Initializes an empty model.
     */
    Model_mml() = default;

    /**
     * @brief Constructor for Model_mml with initial nodes.
     * 
     * @param initialNodes A vector of unique pointers to Node objects to initialize the model with.
     */
    explicit Model_mml(vector<unique_ptr<Node>>&& initialNodes, std::unordered_map<string, GeneralDataTypes>& iomap, std::vector<std::string> inputs, std::vector<std::string> outputs)
        : nodes(move(initialNodes)), iomap(iomap), inputs(inputs), outputs(outputs) {}

    /**
     * @brief Adds a node to the model graph.
     * 
     * @param node A unique pointer to a Node object to be added to the graph.
     */
    void addNode(unique_ptr<Node> node) {
        nodes.push_back(move(node));
    }

    /**
     * @brief Runs inference on the graph.
     * 
     * @param tensor A reference to the input data for inference.
     * @return GeneralDataTypes The result of the inference.
     */
    std::unordered_map<string, GeneralDataTypes> infer(const std::unordered_map<string, GeneralDataTypes>& inputs) override;

private:
    // Nodes in the graph
    vector<unique_ptr<Node>> nodes;

    // Map of inputs and outputs
    std::unordered_map<string, GeneralDataTypes> iomap;

    // Inputs and outputs
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};