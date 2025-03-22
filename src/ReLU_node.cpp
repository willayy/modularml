#include "include/ReLU_node.hpp"

ReLUNode::ReLUNode(std::string X, std::string Y): X(X), Y(Y) {}

ReLUNode::ReLUNode(const json& node) {
  if (node.contains("inputs") && node["inputs"].is_array()) {
    X = node["inputs"][0];
  }

  if (node.contains("outputs") && node["outputs"].is_array()) {
    Y = node["outputs"][0];
  }
}

void ReLUNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
      throw std::runtime_error("ReluNode: Input tensor X not found in iomap");
  }
  
  const GeneralDataTypes& x_tensor = x_it->second;

  std::visit([&](const auto& x_ptr) {
    using TensorPtr = std::decay_t<decltype(x_ptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    if constexpr (!is_in_variant_v<ValueType, T>) {
      throw std::runtime_error("ReluNode: Unsupported data type for tensor X");
    } else {
      auto y_it = iomap.find(Y);
      if (y_it == iomap.end()) {
        throw std::runtime_error("ReluNode: Output tensor Y not found in iomap");
      }

      auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);

      Arithmetic_mml<ValueType> arithmetic;
      arithmetic.elementwise(x_ptr, [](ValueType x) -> ValueType { return x > 0 ? x : 0; }, y_ptr);
    }
  }, x_tensor);
}

std::vector<std::string> ReLUNode::getInputs() {
  return {X};
}

std::vector<std::string> ReLUNode::getOutputs() {
  return {Y};
}