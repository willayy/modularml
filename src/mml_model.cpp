#include "include/mml_model.hpp"
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

    // Create output-to-node mapping
    std::unordered_map<std::string, std::vector<shared_ptr<Node>>> nodeMap;
    for (auto& node : nodes) {
        for (const auto& output : node->getOutputs()) {
            nodeMap[output].push_back(node);
        }
    }

    // Calculate in-degrees and build adjacency list
    std::unordered_map<shared_ptr<Node>, int> inDegree;
    std::unordered_map<shared_ptr<Node>, std::vector<shared_ptr<Node>>> adjacentMap;
    
    for (auto& node : nodes) {
        inDegree[node] = 0;
        for (const auto& input : node->getInputs()) {
            auto it = nodeMap.find(input);
            if (it != nodeMap.end()) {
                for (auto& adjacentNode : it->second) {
                    if (adjacentNode != node) {
                        adjacentMap[adjacentNode].push_back(node);
                        inDegree[node]++;
                    }
                }
            }
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
            auto& node = q.front();
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