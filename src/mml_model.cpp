#include "include/mml_model.hpp"

std::unordered_map<string, GeneralDataTypes> Model_mml::infer(const std::unordered_map<string, GeneralDataTypes>& inputs) {
    if (nodes.empty()) {
        throw runtime_error("ComputeGraph has no nodes.");
    }

    // Set input tensors
    for (const auto& [name, tensor] : inputs) {
        iomap[name] = tensor;
    }

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
 
            //if (!node.areInputsFilled()) {
                //continue;  // Wait for inputs
            //}
 
            // Execute node
            //node.forward();
            executed.insert(&node);
        }
 
        // If no new nodes executed, it means there are unresolved dependencies
        if (executed.size() == prevExecutedSize) {
            throw runtime_error("Not all nodes could be executed; some input tensors remain unfilled.");
        }
    }
 
    // Get output(s)
    std::unordered_map<string, GeneralDataTypes> returnMap;
    for (const auto& name: outputs) {
        if (iomap.find(name) != iomap.end()) {
            returnMap[name] = iomap[name];
        }
    }

    return returnMap;
}