#include "include/mml_parser.hpp"
#include "include/mml_tensor.hpp"
#include "include/ReLU_node.hpp"
#include "include/Swish_node.hpp"
#include "include/TanH_node.hpp"
#include "include/gemm_node.hpp"
#include "include/reshape_node.hpp"
#include "include/flatten_node.hpp"
#include "include/Dropout_node.hpp"
#include "include/conv_node.hpp"
#include "include/mml_model.hpp"

// Helper function: to map the tensors
std::unordered_map<std::string, GeneralDataTypes> mapTensors(const json& graph) {
  std::unordered_map<std::string, GeneralDataTypes> tensorMap;
  
  // First look for already initialized inputs
  if (graph.contains("initializer") && graph["initializer"].is_array()) {
    for (const auto& init: graph["initializer"]) {
      std::string initName = init["name"];
      int dataType = init["dataType"];
      
      std::vector<int> dims;
      for (const auto& el : init["dims"]) {
        dims.push_back(std::stoi(el.get<std::string>()));
      }
      array_mml shapeArray(dims);

      // Need to handle more data types
      if (dataType == 1) {
        std::vector<float> data = init["floatData"].get<std::vector<float>>();
        array_mml dataArray(data);
        tensorMap[initName] = std::make_shared<Tensor_mml<float>>(shapeArray, dataArray);
      } else if (dataType == 7) {
        std::vector<int64_t> data;
        for (const auto& el : init["int64Data"]) {
          data.push_back(std::stoll(el.get<std::string>()));
        }
        array_mml dataArray(data);
        tensorMap[initName] = std::make_shared<Tensor_mml<int64_t>>(shapeArray, dataArray);
      } else {
        throw std::runtime_error("Currently unsupported data type: " + std::to_string(dataType));
      }
    }
  }

  // Then look for inputs
  if (graph.contains("input") && graph["input"].is_array()) {
    for (const auto& input: graph["input"]) {
      std::string inputName = input["name"];
      int dataType = input["type"]["tensorType"]["elemType"];
      
      std::vector<int> dims;
      for (const auto& dim : input["type"]["tensorType"]["shape"]["dim"]) {
        dims.push_back(std::stoi(dim["dimValue"].get<std::string>()));
      }
      array_mml shapeArray(dims);
      
      if (dataType == 1) {
        tensorMap[inputName] = std::make_shared<Tensor_mml<float>>(shapeArray);
      } else if (dataType == 7) {
        tensorMap[inputName] = std::make_shared<Tensor_mml<int64_t>>(shapeArray);
      } else {
        throw std::runtime_error("Currently unsupported data type: " + std::to_string(dataType));
      }
    }
  }

  // Then look for outputs
  if (graph.contains("output") && graph["output"].is_array()) {
    for (const auto& output: graph["output"]) {
      std::string outputName = output["name"];
      int dataType = output["type"]["tensorType"]["elemType"];
      
      std::vector<int> dims;
      for (const auto& dim : output["type"]["tensorType"]["shape"]["dim"]) {
        dims.push_back(std::stoi(dim["dimValue"].get<std::string>()));
      }
      array_mml shapeArray(dims);
      
      if (dataType == 1) {
        tensorMap[outputName] = std::make_shared<Tensor_mml<float>>(shapeArray);
      } else if (dataType == 7) {
        tensorMap[outputName] = std::make_shared<Tensor_mml<int64_t>>(shapeArray);
      } else {
        throw std::runtime_error("Currently unsupported data type: " + std::to_string(dataType));
      }
    }
  }

  // Then look for valueInfo
  if (graph.contains("valueInfo") && graph["valueInfo"].is_array()) {
    for (const auto& valueInfo: graph["valueInfo"]) {
      std::string valueInfoName = valueInfo["name"];
      int dataType = valueInfo["type"]["tensorType"]["elemType"];
      
      std::vector<int> dims;
      for (const auto& dim : valueInfo["type"]["tensorType"]["shape"]["dim"]) {
        dims.push_back(std::stoi(dim["dimValue"].get<std::string>()));
      }
      array_mml shapeArray(dims);
      
      if (dataType == 1) {
        tensorMap[valueInfoName] = std::make_shared<Tensor_mml<float>>(shapeArray);
      } else if (dataType == 7) {
        tensorMap[valueInfoName] = std::make_shared<Tensor_mml<int64_t>>(shapeArray);
      } else {
        throw std::runtime_error("Currently unsupported data type: " + std::to_string(dataType));
      }
    }
  }

  return tensorMap;
}

// Helper function: to construct the nodes
std::vector<shared_ptr<Node>> constructNodes(const json& graph) {
  std::vector<shared_ptr<Node>> nodes;
  
  // Look for nodes
  if (graph.contains("node") && graph["node"].is_array()) {
    for (const auto& node: graph["node"]) {
      std::string opType = node["opType"];
      
      if (opType == "Relu") {
        nodes.push_back(std::make_shared<ReLUNode>(node));
      } else if (opType == "TanH") {
        nodes.push_back(std::make_shared<TanHNode>(node));
      } else if (opType == "HardSwish") {
        nodes.push_back(std::make_shared<SwishNode>(node));
      } else if (opType == "Gemm") {
        nodes.push_back(std::make_shared<GemmNode>(node));
      } else if (opType == "Reshape") {
        nodes.push_back(std::make_shared<reshapeNode>(node));
      } else if (opType == "Flatten") {
        nodes.push_back(std::make_shared<FlattenNode>(node));
      } else if (opType == "Dropout") {
        nodes.push_back(std::make_shared<DropoutNode>(node));
      } else if (opType == "Conv") {
        nodes.push_back(std::make_shared<ConvNode>(node));
      } else {
        throw std::runtime_error("Currently unsupported operation type: " + opType);
      }
    }
  }

  return nodes;
}

// Helper function: Get inputs
std::vector<std::string> getInputs(const json& graph) {
  std::vector<std::string> inputs;
  
  if (graph.contains("input") && graph["input"].is_array()) {
    for (const auto& input: graph["input"]) {
      inputs.push_back(input["name"]);
    }
  }
  
  return inputs;
}

// Helper function: Get outputs
std::vector<std::string> getOutputs(const json& graph) {
  std::vector<std::string> outputs;
  
  if (graph.contains("output") && graph["output"].is_array()) {
    for (const auto& output: graph["output"]) {
      outputs.push_back(output["name"]);
    }
  }
  
  return outputs;
}

unique_ptr<Model> Parser_mml::parse(const json& data) const {
    // Get the graph
    json graph = data["graph"];
    
    // Get the tensors
    std::unordered_map<std::string, GeneralDataTypes> iomap = mapTensors(graph);
    
    // Construct the nodes
    std::vector<shared_ptr<Node>> nodes = constructNodes(graph);

    // Get the inputs
    std::vector<std::string> inputs = getInputs(graph);

    // Get the outputs
    std::vector<std::string> outputs = getOutputs(graph);
    
    // Create the model
    return std::make_unique<Model_mml>(nodes, iomap, inputs, outputs);
}