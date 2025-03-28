#include "backend/mml_model.hpp"
#include <set>
#include <queue>

std::unordered_map<string, GeneralDataTypes> Model_mml::infer(const std::unordered_map<string, GeneralDataTypes>& inputs) {
    if (nodes.empty()) {
        throw runtime_error("ComputeGraph has no nodes.");
    }

    // Perform topological sort
    vector<vector<shared_ptr<Node>>> topoLayers = topologicalSort();

    // Copy the iomap to avoid modifying the original
    std::unordered_map<string, GeneralDataTypes> local_iomap = iomap;

    // Set input tensors
    for (const auto& [name, tensor] : inputs) {
        local_iomap[name] = tensor;
    }

    for (const auto& layer : topoLayers) {
        for (const auto& node : layer) {
            node->forward(local_iomap);
        }
    }
 
    // Get output(s)
    std::unordered_map<string, GeneralDataTypes> returnMap;
    for (const auto& name: outputs) {
        if (local_iomap.find(name) != local_iomap.end()) {
            returnMap[name] = local_iomap[name];
        }
    }

    return returnMap;
}

vector<vector<shared_ptr<Node>>> Model_mml::topologicalSort() {
    if (nodes.empty()) {
        throw runtime_error("ComputeGraph has no nodes.");
    }

    // Create output-to-node mapping (which node produces which tensor)
    std::unordered_map<std::string, shared_ptr<Node>> producerMap;
    for (auto& node : nodes) {
        for (const auto& output : node->getOutputs()) {
            producerMap[output] = node;
        }
    }

    // Calculate in-degrees and build adjacency list
    std::unordered_map<shared_ptr<Node>, int> inDegree;
    std::unordered_map<shared_ptr<Node>, std::vector<shared_ptr<Node>>> adjacentMap;
    
    // Initialize in-degree to zero for all nodes
    for (auto& node : nodes) {
        inDegree[node] = 0;
    }
    
    // Build the adjacency list: if node B consumes output from node A, add A → B edge
    for (auto& consumerNode : nodes) {
        for (const auto& input : consumerNode->getInputs()) {
            auto producerIt = producerMap.find(input);
            if (producerIt != producerMap.end()) {
                shared_ptr<Node> producerNode = producerIt->second;
                if (producerNode != consumerNode) { // Avoid self-loops
                    adjacentMap[producerNode].push_back(consumerNode);
                    inDegree[consumerNode]++;
                }
            }
            // Inputs that aren't in producerMap are external inputs or initializers
        }
    }

    // Queue nodes with zero in-degree
    std::queue<shared_ptr<Node>> q;
    for (auto& node : nodes) {
        if (inDegree[node] == 0) {
            q.push(node);
        }
    }

    // Perform topological sort
    vector<vector<shared_ptr<Node>>> layers;
    int processedCount = 0;
    
    while (!q.empty()) {
        int size = q.size();
        vector<shared_ptr<Node>> currentLayer;
        currentLayer.reserve(size);
        
        for (int i = 0; i < size; i++) {
            auto node = q.front();
            q.pop();
            
            currentLayer.push_back(node);
            processedCount++;
            
            for (auto& adjacentNode : adjacentMap[node]) {
                if (--inDegree[adjacentNode] == 0) {
                    q.push(adjacentNode);
                }
            }
        }
        layers.push_back(currentLayer);
    }

    // Check for cycles
    if (processedCount != nodes.size()) {
        throw runtime_error("ComputeGraph has a cycle.");
    }

    return layers;
}