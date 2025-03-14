#include "include/mml_parser.hpp"
#include "include/mml_tensor.hpp"
#include "include/ReLU_node.hpp"
#include "include/Swish_node.hpp"
#include "include/TanH_node.hpp"
#include "include/gemm_node.hpp"
#include "include/mml_model.hpp"

// Helper function: to create a GEMM node
std::unique_ptr<Node> makeGemm(const GeneralDataTypes& a, const GeneralDataTypes& b, 
  const GeneralDataTypes& y, const std::optional<GeneralDataTypes>& c = std::nullopt,
  float alpha = 1.0f, float beta = 1.0f, int transA = 0, int transB = 0) {
  return std::visit([&](const auto& aptr) -> std::unique_ptr<Node> {
    // Get the type of the tensor element
    using TensorPtr = std::decay_t<decltype(aptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    // Get the output tensors with the same type as the input
    auto bptr = std::get<std::shared_ptr<Tensor<ValueType>>>(b);
    auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
    
    // Handle optional C tensor
    std::optional<std::shared_ptr<Tensor<ValueType>>> cptr = std::nullopt;
    if (c.has_value()) {
      cptr = std::get<std::shared_ptr<Tensor<ValueType>>>(c.value());
    }
    
    // Create a GemmNode with the appropriate type
    return std::make_unique<GemmNode<ValueType>>(
      aptr, bptr, yptr, cptr, alpha, beta, transA, transB);
  }, a);
}

// Helper function: to create a ReLU node
std::unique_ptr<Node> makeRelu(const GeneralDataTypes& x, const GeneralDataTypes& y) {
  return std::visit([&y](const auto& xptr) -> std::unique_ptr<Node> {
      // Get the type of the tensor element
      using TensorPtr = std::decay_t<decltype(xptr)>;
      using TensorType = typename TensorPtr::element_type;
      using ValueType = typename TensorType::value_type;
      
      // Get the output tensor with the same type as the input
      auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
      
      // Create a ReLUNode with the appropriate type
      return std::make_unique<ReLUNode<ValueType>>(xptr, yptr);
  }, x);
}

// Helper function: to create a Swish node
std::unique_ptr<Node> makeSwish(const GeneralDataTypes& x, const GeneralDataTypes& y) {
  return std::visit([&y](const auto& xptr) -> std::unique_ptr<Node> {
      // Get the type of the tensor element
      using TensorPtr = std::decay_t<decltype(xptr)>;
      using TensorType = typename TensorPtr::element_type;
      using ValueType = typename TensorType::value_type;
      
      // Get the output tensor with the same type as the input
      auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
      
      // Create a SwishNode with the appropriate type
      return std::make_unique<SwishNode<ValueType>>(xptr, yptr);
  }, x);
}

// Helper function: to create a TanH node
std::unique_ptr<Node> makeTanH(const GeneralDataTypes& x, const GeneralDataTypes& y) {
  return std::visit([&y](const auto& xptr) -> std::unique_ptr<Node> {
      // Get the type of the tensor element
      using TensorPtr = std::decay_t<decltype(xptr)>;
      using TensorType = typename TensorPtr::element_type;
      using ValueType = typename TensorType::value_type;
      
      // Get the output tensor with the same type as the input
      auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
      
      // Create a TanHNode with the appropriate type
      return std::make_unique<TanHNode<ValueType>>(xptr, yptr);
  }, x);
}

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
      
      if (opType == "Relu") {
        nodes.push_back(makeRelu(inputs[0], outputs[0]));
      } else if (opType == "TanH") {
        nodes.push_back(makeTanH(inputs[0], outputs[0]));
      } else if (opType == "Swish") {
        nodes.push_back(makeSwish(inputs[0], outputs[0]));
      } else {
        throw std::runtime_error("Currently unsupported operation type: " + opType);
      }
    }
  }

  return nodes;
}

unique_ptr<Model> Parser_mml::parse(const json& data) const {
    // Get the graph
    json graph = data["graph"];
    
    // Get the tensors
    std::unordered_map<std::string, GeneralDataTypes> tensors = mapTensors(graph);
    
    // Construct the nodes
    std::vector<unique_ptr<Node>> nodes = constructNodes(graph, tensors);
    
    // Create the model
    return std::make_unique<Model_mml>(move(nodes));
}