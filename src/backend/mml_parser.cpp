#include "backend/mml_parser.hpp"
#include "backend/mml_model.hpp"
#include "backend/parser_helper.hpp"
#include "nodes/add.hpp"
#include "nodes/avg_pooling.hpp"
#include "nodes/constant.hpp"
#include "nodes/conv.hpp"
#include "nodes/dropout.hpp"
#include "nodes/elu.hpp"
#include "nodes/flatten.hpp"
#include "nodes/gelu.hpp"
#include "nodes/gemm.hpp"
#include "nodes/leaky_relu.hpp"
#include "nodes/log_softmax.hpp"
#include "nodes/lrn.hpp"
#include "nodes/max_pooling.hpp"
#include "nodes/relu.hpp"
#include "nodes/reshape.hpp"
#include "nodes/sigmoid.hpp"
#include "nodes/tanh.hpp"
#include "nodes/swish.hpp"

// Helper std::function: to map the tensors
std::unordered_map<std::string, GeneralDataTypes>
mapTensors(const nlohmann::json &graph) {
  std::unordered_map<std::string, GeneralDataTypes> tensorMap;

  // First look for already initialized inputs
  if (graph.contains("initializer") && graph["initializer"].is_array()) {
    for (const auto &init : graph["initializer"]) {
      std::string initName = init["name"];
      int dataType = init["dataType"];

      // Need to handle more data types
      switch (dataType) {
      case 1: // FLOAT
        tensorMap[initName] = ParserHelper::handle_tensor<float>(init);
        break;
      case 2: // UINT8
        tensorMap[initName] = ParserHelper::handle_tensor<uint8_t>(init);
        break;
      case 3: // INT8
        tensorMap[initName] = ParserHelper::handle_tensor<int8_t>(init);
        break;
      case 4: // UINT16
        tensorMap[initName] = ParserHelper::handle_tensor<uint16_t>(init);
        break;
      case 5: // INT16
        tensorMap[initName] = ParserHelper::handle_tensor<int16_t>(init);
        break;
      case 6: // INT32
        tensorMap[initName] = ParserHelper::handle_tensor<int32_t>(init);
        break;
      case 7: // INT64
        tensorMap[initName] = ParserHelper::handle_tensor<int64_t>(init);
        break;
      case 9: // BOOL
        tensorMap[initName] = ParserHelper::handle_tensor<bool>(init);
        break;
      case 11: // DOUBLE
        tensorMap[initName] = ParserHelper::handle_tensor<double>(init);
        break;
      case 12: // UINT32
        tensorMap[initName] = ParserHelper::handle_tensor<uint32_t>(init);
        break;
      case 13: // UINT64
        tensorMap[initName] = ParserHelper::handle_tensor<uint64_t>(init);
        break;
      default:
        throw std::runtime_error("Currently unsupported data type: " +
                                 std::to_string(dataType));
      }
    }
  }

  return tensorMap;
}

// Helper std::function: to construct the nodes
std::vector<std::shared_ptr<Node>> constructNodes(const nlohmann::json &graph) {
  std::vector<std::shared_ptr<Node>> nodes;

  // Look for nodes
  if (graph.contains("node") && graph["node"].is_array()) {
    for (const auto &node : graph["node"]) {
      std::string opType = node["opType"];

      // This will later be switched to a map
      if (opType == "Add") {
        nodes.push_back(std::make_shared<AddNode>(node));
      } else if (opType == "AveragePool") {
        nodes.push_back(std::make_shared<AvgPoolingNode_mml>(node));
      } else if (opType == "Constant") {
        nodes.push_back(std::make_shared<ConstantNode>(node));
      } else if (opType == "Conv") {
        nodes.push_back(std::make_shared<ConvNode>(node));
      } else if (opType == "Dropout") {
        nodes.push_back(std::make_shared<DropoutNode>(node));
      } else if (opType == "Elu") {
        nodes.push_back(std::make_shared<ELUNode>(node));
      } else if (opType == "Flatten") {
        nodes.push_back(std::make_shared<FlattenNode>(node));
      } else if (opType == "Gelu") {
        nodes.push_back(std::make_shared<GeluNode>(node));
      } else if (opType == "Gemm") {
        nodes.push_back(std::make_shared<GemmNode>(node));
      } else if (opType == "LeakyRelu") {
        nodes.push_back(std::make_shared<LeakyReLUNode>(node));
      } else if (opType == "LogSoftmax") {
        nodes.push_back(std::make_shared<LogSoftMaxNode>(node));
      } else if (opType == "LRN") {
        nodes.push_back(std::make_shared<LRNNode_mml>(node));
      } else if (opType == "MaxPool") {
        nodes.push_back(std::make_shared<MaxPoolingNode_mml>(node));
      } else if (opType == "Relu") {
        nodes.push_back(std::make_shared<ReLUNode>(node));
      } else if (opType == "Reshape") {
        nodes.push_back(std::make_shared<reshapeNode>(node));
      } else if (opType == "Sigmoid") {
        nodes.push_back(std::make_shared<SigmoidNode>(node));
      } else if (opType == "Swish") {
        nodes.push_back(std::make_shared<SwishNode>(node));
      } else if (opType == "Tanh") {
        nodes.push_back(std::make_shared<TanHNode>(node));
      } else {
        throw std::runtime_error("Currently unsupported operation type: " +
                                 opType);
      }
    }
  }

  return nodes;
}

// Helper std::function: Get inputs
std::vector<std::string> getInputs(const nlohmann::json &graph) {
  std::vector<std::string> inputs;

  if (graph.contains("input") && graph["input"].is_array()) {
    for (const auto &input : graph["input"]) {
      inputs.push_back(input["name"]);
    }
  }

  return inputs;
}

// Helper std::function: Get outputs
std::vector<std::string> getOutputs(const nlohmann::json &graph) {
  std::vector<std::string> outputs;

  if (graph.contains("output") && graph["output"].is_array()) {
    for (const auto &output : graph["output"]) {
      outputs.push_back(output["name"]);
    }
  }

  return outputs;
}

std::unique_ptr<Model> Parser_mml::parse(const nlohmann::json &data) const {
  // Get the graph
  nlohmann::json graph = data["graph"];

  // Get the tensors
  std::unordered_map<std::string, GeneralDataTypes> iomap = mapTensors(graph);

  // Construct the nodes
  std::vector<std::shared_ptr<Node>> nodes = constructNodes(graph);

  // Get the inputs
  std::vector<std::string> inputs = getInputs(graph);

  // Get the outputs
  std::vector<std::string> outputs = getOutputs(graph);

  // Create the model
  return std::make_unique<Model_mml>(nodes, iomap, inputs, outputs);
}