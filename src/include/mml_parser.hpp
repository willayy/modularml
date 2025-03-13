#pragma once

#include "a_data_parser.hpp"

std::unordered_map<std::string, GeneralDataTypes> mapTensors(const json& nodes) {
  std::unordered_map<std::string, GeneralDataTypes> tensorMap;
  
  // First look for already initialized inputs
  for (const auto& node_json: nodes) {
    if (node_json.contains("initializers") && node_json["initializers"].is_array()) {
      for (const auto& init: node_json["initializers"]) {
        std::string initName = init["name"];
        if (tensorMap.find(initName) != tensorMap.end()) {
          std::string typeName = init["data_type"];

          std::vector<int> shapeVec = init["shape"].get<std::vector<int>>();
          array_mml<int> shapeArray(shapeVec);

          // For valueArray we should read it from binary in the future!
          if (typeName == "FLOAT") {
            auto valueArray = flattenToArrayMml<float>(init["values"]);
            tensorMap[initName] = std::make_shared<Tensor_mml<float>>(shapeArray,valueArray);
          } else if (typeName == "DOUBLE") {
            auto valueArray = flattenToArrayMml<double>(init["values"]);
            tensorMap[initName] = std::make_shared<Tensor_mml<double>>(shapeArray, valueArray);
          } else if (typeName == "INT32") {
            auto valueArray = flattenToArrayMml<int32_t>(init["values"]);
            tensorMap[initName] = std::make_shared<Tensor_mml<int32_t>>(shapeArray, valueArray);
          } else if (typeName == "INT64") {
            auto valueArray = flattenToArrayMml<int64_t>(init["values"]);
            tensorMap[initName] = std::make_shared<Tensor_mml<int64_t>>(shapeArray, valueArray);
          } else {
            // Templates are resolved at compilation so need to use if/else or switch.
            throw std::runtime_error("Currently supported data type: " + typeName);
          }
        }
      }
    }
  }

  // Then we make empty construction for unique inputs/outputs
  for (const auto& node_json: nodes) {
    if (node_json.contains("inputs") && node_json["inputs"].is_array()) {
      for (const auto& input: node_json["inputs"]) {
        std::string inputName = input["name"];
        if (tensorMap.find(inputName) != tensorMap.end()) {
          tensorMap[inputName] = nullptr;
        }
      }
    }
    if (node_json.contains("outputs") && node_json["outputs"].is_array()) {
      for (const auto& output: node_json["outputs"]) {
        std::string outputName = output["name"];
        if (tensorMap.find(outputName) != tensorMap.end()) {
          tensorMap[outputName] = nullptr;
        }
      }
    }
  }
}

class Parser_mml: public DataParser {
  public:
    Parser_mml() = default;
    
    unique_ptr<Model> parse(const json& data) const {
      vector<unique_ptr<Node>> nodes;
      if (data.contains("nodes") && data["nodes"].is_array()) {
        for (const auto& node_json: data["nodes"]) {
          
        }
      }
    }
}
