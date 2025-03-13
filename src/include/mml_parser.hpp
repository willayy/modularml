#pragma once

#include "a_data_parser.hpp"
#include "mml_tensor.hpp"


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
std::vector<unique_ptr<Node>> constructNodes(const json& graph, const std::unordered_map<std::string, GeneralDataTypes> tensorMap) {
  std::vector<unique_ptr<Node>> nodes;
  
  // Look for nodes
  if (graph.contains("node") && graph["node"].is_array()) {
    for (const auto& node: graph["node"]) {
      std::string opType = node["opType"];
      
      // Get the input tensors
      std::vector<GeneralDataTypes> inputs;
      for (const auto& input: node["input"]) {
        inputs.push_back(tensorMap.at(input.get<std::string>()));
      }
      
      // Get the output tensors
      std::vector<GeneralDataTypes> outputs;
      for (const auto& output: node["output"]) {
        outputs.push_back(tensorMap.at(output.get<std::string>()));
      }
      
      // Create the node
      if (opType == "Add") {
        nodes.push_back(std::make_unique<>);
      } else if (opType == "Sub") {
        nodes.push_back(std::make_unique<>);
      } else if (opType == "Mul") {
        nodes.push_back(std::make_unique<>);
      } else if (opType == "Div") {
        nodes.push_back(std::make_unique<>);
      } else {
        throw std::runtime_error("Currently unsupported operation type: " + opType);
      }
    }
  }

  return nodes;
}

class Parser_mml: public DataParser {
  public:
    Parser_mml() = default;
    
    unique_ptr<Model> parse(const json& data) const {
      //Get the graph
      json graph = data["graph"];

      // Get the tensors
      std::unordered_map<std::string, GeneralDataTypes> tensors = mapTensors(graph);




    }
}
