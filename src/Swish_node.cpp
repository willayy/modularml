#include "include/Swish_node.hpp"

SwishNode::SwishNode(std::string X, std::string Y): X(X), Y(Y) {}

SwishNode::SwishNode(const json& node) {
  if (node.contains("inputs") && node["inputs"].is_array()) {
    X = node["inputs"][0];
  }

  if (node.contains("outputs") && node["outputs"].is_array()) {
    Y = node["outputs"][0];
  }
}

void SwishNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
      throw std::runtime_error("SwishNode: Input tensor X not found in iomap");
  }
  
  const GeneralDataTypes& x_tensor = x_it->second;

  std::visit([&](const auto& x_ptr) {
    using TensorPtr = std::decay_t<decltype(x_ptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    if constexpr (!is_in_variant_v<ValueType, T>) {
      throw std::runtime_error("SwishNode: Unsupported data type for tensor X");
    } else {
      auto y_it = iomap.find(Y);
      if (y_it == iomap.end()) {
        throw std::runtime_error("SwishNode: Output tensor Y not found in iomap");
      }

      auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);

      Arithmetic_mml<ValueType> arithmetic;
      arithmetic.elementwise(x_ptr, [](ValueType x) -> ValueType {
        ValueType sigmoid_x = static_cast<ValueType>(1) / (static_cast<ValueType>(1) + exp(-x));
        return x * sigmoid_x; }, y_ptr);
    }
  }, x_tensor);
}

std::vector<std::string> SwishNode::getInputs() {
  return {X};
}

std::vector<std::string> SwishNode::getOutputs() {
  return {Y};
}