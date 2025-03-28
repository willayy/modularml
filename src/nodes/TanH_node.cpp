#include "nodes/TanH_node.hpp"

TanHNode::TanHNode(std::string X, std::string Y): X(X), Y(Y) {}

TanHNode::TanHNode(const json& node) {
  if (node.contains("input") && node["input"].is_array()) {
    X = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }
}

void TanHNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
      throw std::runtime_error("TanHNode: Input tensor X not found in iomap");
  }
  
  const GeneralDataTypes& x_tensor = x_it->second;

  std::visit([&](const auto& x_ptr) {
    using TensorPtr = std::decay_t<decltype(x_ptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;
    
    if constexpr (!is_in_variant_v<ValueType, T>) {
      throw std::runtime_error("TanHNode: Unsupported data type for tensor X");
    } else {
      auto y_it = iomap.find(Y);
      if (y_it == iomap.end()) {
        // Create output tensor if it doesn't exist
        auto y_ptr = x_ptr->copy();
        // No need to fill with zeros as the elementwise function will overwrite the values
        iomap[Y] = y_ptr;
        y_it = iomap.find(Y);
      } else if (!std::holds_alternative<std::shared_ptr<Tensor<ValueType>>>(y_it->second)) {
        throw std::runtime_error("TanHNode: Output tensor Y has incorrect type");
      }

      auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);

      Arithmetic_mml<ValueType> arithmetic;
      arithmetic.elementwise(x_ptr, [](ValueType x) -> ValueType { return tanh(x); }, y_ptr);
    }
  }, x_tensor);
}

std::vector<std::string> TanHNode::getInputs() {
  return {X};
}

std::vector<std::string> TanHNode::getOutputs() {
  return {Y};
}