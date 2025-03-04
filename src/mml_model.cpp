#include "include/mml_model.hpp"

GeneralDataTypes Model_mml::infer(GeneralDataTypes& tensor) {
    if (nodes.empty()) {
        throw runtime_error("ComputeGraph has no nodes.");
    }

    // Set input tensor to the first node (believe it will always be the first node, but parser can enforce this)
    Node& firstNode = *nodes.front();
    firstNode.setInput(tensor);

    // Track executed nodes
    unordered_set<Node*> executed;

    // Loop until all nodes are executed
    while (executed.size() < nodes.size()) {
        size_t prevExecutedSize = executed.size();
 
        for (const auto& nodePtr : nodes) {
            Node& node = *nodePtr;

            if (executed.count(&node)) {
                continue;  // Already executed
            }
 
            if (!node.areInputsFilled()) {
                continue;  // Wait for inputs
            }
 
            // Execute node
            node.forward();
            executed.insert(&node);
        }
 
        // If no new nodes executed, it means there are unresolved dependencies
        if (executed.size() == prevExecutedSize) {
            throw runtime_error("Not all nodes could be executed; some input tensors remain unfilled.");
        }
    }
 
    // Get output from the last node
    if (const Node& lastNode = *nodes.back(); lastNode.areOutputsFilled()) {
        return lastNode.getOutput();
    }
 
    throw runtime_error("No valid output found in the compute graph.");
 }