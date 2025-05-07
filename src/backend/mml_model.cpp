#include "backend/mml_model.hpp"

#include <iostream>
// IWYU pragma: no_include <__ostream/basic_ostream.h>
#include <ostream>  // IWYU pragma: keep
#include <queue>
#include <stdexcept>
#include <typeinfo>
#include <variant>

#include <thread>
#include "utility/inference_progress_spinner.hpp"

std::unordered_map<std::string, GeneralDataTypes> Model_mml::infer(
    const std::unordered_map<std::string, GeneralDataTypes> &inputs) {
  
#if defined(VERBOSE_INFERENCE)
  std::cout << "==== Starting Inference ====" << std::endl;
#endif
  if (nodes.empty()) {
    throw std::runtime_error("ComputeGraph has no nodes.");
  }

  // Perform topological sort
  std::vector<std::vector<std::shared_ptr<Node>>> topoLayers =
      topologicalSort();
  //std::cout << "Topological layers: " << topoLayers.size() << std::endl;

  // Create a deep copy of iomap
  std::unordered_map<std::string, GeneralDataTypes> local_iomap;
  for (const auto &[name, tensor] : iomap) {
    std::visit([&](auto &&arg) {
      // Deep copy
      local_iomap[name] = arg->copy();
    },
               tensor);
  }

  // Set input tensors
  for (const auto &[name, tensor] : inputs) {
    //std::cout << "Setting input: " << name << std::endl;
    std::visit([&](auto &&arg) {
      // Deep copy
      local_iomap[name] = arg->copy();
    },
               tensor);
  }
#if defined(VERBOSE_INFERENCE)
  std::thread spinner(inference_spinner_function, topoLayers.size());
#endif
  // Process each layer
  try {
    for (size_t layer_idx = 0; layer_idx < topoLayers.size(); ++layer_idx) {
      const auto &layer = topoLayers[layer_idx];
      //std::cout << "Processing layer " << layer_idx << " with " << layer.size()
      //          << " nodes" << std::endl;

      current_layer_idx.store(layer_idx);

      for (size_t node_idx = 0; node_idx < layer.size(); ++node_idx) {
        const auto &node = layer[node_idx];
        std::string nodeType = typeid(*node).name();  // Get node type
        //std::cout << "  Processing node " << node_idx << " (type: " << nodeType
        //          << ")" << std::endl;

        try {
          node->forward(local_iomap);
          //std::cout << "  Node " << nodeType << " processed successfully"
          //        << std::endl;
        } catch (const std::out_of_range &e) {
          std::cerr << "*** Out of range error in node " << node_idx
                    << " (type: " << nodeType << "): " << e.what() << std::endl;

          // Print node inputs and outputs
          std::cout << "  Node inputs: ";
          for (const auto &input : node->getInputs()) {
            std::cout << input << " ";
          }
          std::cout << std::endl;

          std::cout << "  Node outputs: ";
          for (const auto &output : node->getOutputs()) {
            std::cout << output << " ";
          }
          std::cout << std::endl;

          // Rethrow so the test catches it
          throw;
        } catch (const std::exception &e) {
          std::cerr << "*** Error in node " << node_idx
                    << " (type: " << nodeType << "): " << e.what() << std::endl;
          throw;
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Exception during inference: " << e.what() << std::endl;
    throw;
  }

  // Get output(s)
  std::unordered_map<std::string, GeneralDataTypes> returnMap;
  for (const auto &name : outputs) {
    if (local_iomap.find(name) != local_iomap.end()) {
      returnMap[name] = local_iomap[name];
    }
  }
  
#if defined(VERBOSE_INFERENCE)
  // Stop progress bar thread
  running_inference.store(false);
  spinner.join();
#endif

  return returnMap;
}

std::vector<std::vector<std::shared_ptr<Node>>> Model_mml::topologicalSort() {
  if (nodes.empty()) {
    throw std::runtime_error("ComputeGraph has no nodes.");
  }

  // Create output-to-node mapping (which node produces which tensor)
  std::unordered_map<std::string, std::shared_ptr<Node>> producerMap;
  for (auto &node : nodes) {
    for (const auto &output : node->getOutputs()) {
      producerMap[output] = node;
    }
  }

  // Calculate in-degrees and build adjacency list
  std::unordered_map<std::shared_ptr<Node>, int> inDegree;
  std::unordered_map<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>>
      adjacentMap;

  // Initialize in-degree to zero for all nodes
  for (auto &node : nodes) {
    inDegree[node] = 0;
  }

  // Build the adjacency list: if node B consumes output from node A, add A → B
  // edge
  for (auto &consumerNode : nodes) {
    for (const auto &input : consumerNode->getInputs()) {
      auto producerIt = producerMap.find(input);
      if (producerIt != producerMap.end()) {
        std::shared_ptr<Node> producerNode = producerIt->second;
        if (producerNode != consumerNode) {  // Avoid self-loops
          adjacentMap[producerNode].push_back(consumerNode);
          inDegree[consumerNode]++;
        }
      }
      // Inputs that aren't in producerMap are external inputs or initializers
    }
  }

  // Queue nodes with zero in-degree
  std::queue<std::shared_ptr<Node>> q;
  for (auto &node : nodes) {
    if (inDegree[node] == 0) {
      q.push(node);
    }
  }

  // Perform topological sort
  std::vector<std::vector<std::shared_ptr<Node>>> layers;
  int processedCount = 0;

  while (!q.empty()) {
    int size = q.size();
    std::vector<std::shared_ptr<Node>> currentLayer;
    currentLayer.reserve(size);

    for (int i = 0; i < size; i++) {
      auto node = q.front();
      q.pop();

      currentLayer.push_back(node);
      processedCount++;

      for (auto &adjacentNode : adjacentMap[node]) {
        if (--inDegree[adjacentNode] == 0) {
          q.push(adjacentNode);
        }
      }
    }
    layers.push_back(currentLayer);
  }

  // Check for cycles
  if (processedCount != nodes.size()) {
    throw std::runtime_error("ComputeGraph has a cycle.");
  }

  return layers;
}