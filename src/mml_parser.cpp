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

// Helper function: to create a GEMM node
std::unique_ptr<Node> makeGemm(const GeneralDataTypes& a, const GeneralDataTypes& b, 
    const GeneralDataTypes& y, const std::optional<GeneralDataTypes>& c = std::nullopt,
    float alpha = 1.0f, float beta = 1.0f, int transA = 0, int transB = 0) {
    return std::visit([&](const auto& aptr) -> std::unique_ptr<Node> {
        // Get the type of the tensor element
        using TensorPtr = std::decay_t<decltype(aptr)>;
        using TensorType = typename TensorPtr::element_type;
        using ValueType = typename TensorType::value_type;
        
        // GemmNode only supports float, double, int32_t, int64_t, uint32_t, or uint64_t
        if constexpr (std::is_same_v<ValueType, float> || 
                                    std::is_same_v<ValueType, double> || 
                                    std::is_same_v<ValueType, int32_t> || 
                                    std::is_same_v<ValueType, int64_t> ||
                                    std::is_same_v<ValueType, uint32_t> ||
                                    std::is_same_v<ValueType, uint64_t>) {
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
        } else {
            throw std::runtime_error("GemmNode only supports float, double, int32_t, int64_t, uint32_t, or uint64_t types");
        }
    }, a);
}

// Helper function: to create a ReLU node
std::unique_ptr<Node> makeRelu(const GeneralDataTypes& x, const GeneralDataTypes& y) {
    return std::visit([&](const auto& xptr) -> std::unique_ptr<Node> {
            // Get the type of the tensor element
            using TensorPtr = std::decay_t<decltype(xptr)>;
            using TensorType = typename TensorPtr::element_type;
            using ValueType = typename TensorType::value_type;
            
            // ReLUNode only supports float, double, int32_t, int64_t
            if constexpr (std::is_same_v<ValueType, float> || 
                                        std::is_same_v<ValueType, double> || 
                                        std::is_same_v<ValueType, int32_t> || 
                                        std::is_same_v<ValueType, int64_t>) {
                    // Get the output tensor with the same type as the input
                    auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
                    
                    // Create a ReLUNode with the appropriate type
                    return std::make_unique<ReLUNode<ValueType>>(xptr, yptr);
            } else {
                    throw std::runtime_error("ReLUNode only supports float, double, int32_t, int64_t types");
            }
    }, x);
}

// Helper function: to create a Swish node
std::unique_ptr<Node> makeSwish(const GeneralDataTypes& x, const GeneralDataTypes& y) {
  return std::visit([&](const auto& xptr) -> std::unique_ptr<Node> {
      // Get the type of the tensor element
      using TensorPtr = std::decay_t<decltype(xptr)>;
      using TensorType = typename TensorPtr::element_type;
      using ValueType = typename TensorType::value_type;
      
      // SwishNode only supports float and double
      if constexpr (std::is_same_v<ValueType, float> || std::is_same_v<ValueType, double>) {
          // Get the output tensor with the same type as the input
          auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
          
          // Create a SwishNode with the appropriate type
          return std::make_unique<SwishNode<ValueType>>(xptr, yptr);
      } else {
          throw std::runtime_error("SwishNode only supports float and double types");
      }
  }, x);
}

// Helper function: to create a TanH node
std::unique_ptr<Node> makeTanH(const GeneralDataTypes& x, const GeneralDataTypes& y) {
    return std::visit([&](const auto& xptr) -> std::unique_ptr<Node> {
            // Get the type of the tensor element
            using TensorPtr = std::decay_t<decltype(xptr)>;
            using TensorType = typename TensorPtr::element_type;
            using ValueType = typename TensorType::value_type;
            
            // TanHNode only supports float and double
            if constexpr (std::is_same_v<ValueType, float> || std::is_same_v<ValueType, double>) {
                    // Get the output tensor with the same type as the input
                    auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
                    
                    // Create a TanHNode with the appropriate type
                    return std::make_unique<TanHNode<ValueType>>(xptr, yptr);
            } else {
                    throw std::runtime_error("TanHNode only supports float and double types");
            }
    }, x);
}

// Helper function: to create a reshape node
std::unique_ptr<Node> makeReshape(const GeneralDataTypes& data, const GeneralDataTypes& shape, 
                 const GeneralDataTypes& reshaped, int allowzero = 0) {
  return std::visit([&](const auto& dataPtr) -> std::unique_ptr<Node> {
    // Get the type of the tensor element
    using TensorPtr = std::decay_t<decltype(dataPtr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    // reshapeNode supports float, double, int32_t, int64_t, bool, string
    if constexpr (std::is_same_v<ValueType, float> || 
          std::is_same_v<ValueType, double> || 
          std::is_same_v<ValueType, int32_t> || 
          std::is_same_v<ValueType, int64_t> ||
          std::is_same_v<ValueType, bool> ||
          std::is_same_v<ValueType, string>) {
      // Get the shape tensor (must be int64_t type)
      auto shapePtr = std::get<std::shared_ptr<Tensor<int64_t>>>(shape);
      
      // Get the output tensor with the same type as the input
      auto reshapedPtr = std::get<std::shared_ptr<Tensor<ValueType>>>(reshaped);
      
      // Create a reshapeNode with the appropriate type
      return std::make_unique<reshapeNode<ValueType>>(
        dataPtr, shapePtr, reshapedPtr, allowzero);
    } else {
      throw std::runtime_error("reshapeNode only supports float, double, int32_t, int64_t, bool, string types");
    }
  }, data);
}

// Helper function: to create a flatten node
std::unique_ptr<Node> makeFlatten(const GeneralDataTypes& x, const GeneralDataTypes& y, int axis = 1) {
  return std::visit([&](const auto& xptr) -> std::unique_ptr<Node> {
    // Get the type of the tensor element
    using TensorPtr = std::decay_t<decltype(xptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    // Get the output tensor with the same type as the input
    auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y);
    
    // Create a FlattenNode with the appropriate type
    return std::make_unique<FlattenNode<ValueType>>(xptr, yptr, axis);
  }, x);
}

// Helper function: to create droupout node
std::unique_ptr<Node> makeDropout(const GeneralDataTypes& data, const GeneralDataTypes& output,
                 const std::optional<GeneralDataTypes>& mask = std::nullopt,
                 float ratio = 0.5, bool training_mode = false,
                 std::optional<int> seed = std::nullopt) {
  return std::visit([&](const auto& dataPtr) -> std::unique_ptr<Node> {
  // Get the type of the tensor element
  using TensorPtr = std::decay_t<decltype(dataPtr)>;
  using TensorType = typename TensorPtr::element_type;
  using ValueType = typename TensorType::value_type;
  
  // DropoutNode only supports float and double
  if constexpr (std::is_same_v<ValueType, float> || std::is_same_v<ValueType, double>) {
    // Get the output tensor with the same type as the input
    auto outputPtr = std::get<std::shared_ptr<Tensor<ValueType>>>(output);
    
    // Handle optional mask tensor
    std::optional<std::shared_ptr<Tensor<ValueType>>> maskPtr = std::nullopt;
    if (mask.has_value()) {
    maskPtr = std::get<std::shared_ptr<Tensor<ValueType>>>(mask.value());
    }
    
    // Create a DropoutNode with the appropriate type
    return std::make_unique<DropoutNode<ValueType>>(
    dataPtr, outputPtr, maskPtr, ratio, training_mode, seed);
  } else {
    throw std::runtime_error("DropoutNode only supports float and double types");
  }
  }, data);
}

// Helper function: to create conv node
std::unique_ptr<Node> makeConv(const GeneralDataTypes& X, const GeneralDataTypes& W, 
                const GeneralDataTypes& Y,
                array_mml<int> dilations,
                array_mml<int> padding,
                array_mml<int> kernel_shape,
                array_mml<int> stride,
                const std::optional<GeneralDataTypes>& B = std::nullopt,
                int group = 1) {
  return std::visit([&](const auto& xptr) -> std::unique_ptr<Node> {
    // Get the type of the tensor element
    using TensorPtr = std::decay_t<decltype(xptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    // ConvNode only supports double, float, uint
    if constexpr (std::is_same_v<ValueType, double> || 
           std::is_same_v<ValueType, float> || 
           std::is_same_v<ValueType, uint>) {
      // Get the weight and output tensors with the same type as the input
      auto wptr = std::get<std::shared_ptr<Tensor<ValueType>>>(W);
      auto yptr = std::get<std::shared_ptr<Tensor<ValueType>>>(Y);
      
      // Handle optional bias tensor
      std::optional<std::shared_ptr<Tensor<ValueType>>> bptr = std::nullopt;
      if (B.has_value()) {
        bptr = std::get<std::shared_ptr<Tensor<ValueType>>>(B.value());
      }
      
      // Create a ConvNode with the appropriate type
      return std::make_unique<ConvNode<ValueType>>(
        xptr, wptr, yptr, dilations, padding, kernel_shape, stride, bptr, group);
    } else {
      throw std::runtime_error("ConvNode only supports double, float, uint types");
    }
  }, X);
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
      } else if (opType == "HardSwish") {
        nodes.push_back(makeSwish(inputs[0], outputs[0]));
      } else if (opType == "Gemm") {
        // Extract attributes
        // Construct the node
      } else if (opType == "Reshape") {
        // Extract attributes
        // Construct the node
      } else if (opType == "Flatten") {
        // Extract attributes
        // Construct the node
      } else if (opType == "Dropout") {
        // Extract attributes
        // Construct the node
      } else if (opType == "Conv") {
        // Extract attributes
        // Construct the node
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