#include "nodes/leaky_relu.hpp"

LeakyReLUNode::LeakyReLUNode(const std::string &X, const std::string &Y, float alpha)
    : X(X), Y(Y), alpha(alpha) {}

LeakyReLUNode::LeakyReLUNode(const nlohmann::json &node) {
  if (node.contains("input") && node["input"].is_array()) {
    X = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }

  alpha = 1.0f;
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto &attr : node["attribute"]) {
      if (attr["name"] == "alpha") {
        alpha = attr["f"];
      }
    }
  }
}

void LeakyReLUNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
    throw std::runtime_error(
        "LeakyReLUNode: Input tensor X not found in iomap");
  }

  const GeneralDataTypes &x_tensor = x_it->second;

  std::visit(
      [&](const auto &x_ptr) {
        using ValueTypeX =
            typename std::decay_t<decltype(x_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeX, T>) {
          throw std::runtime_error(
              "LeakyReLUNode: Unsupported data type for tensor X");
        } else {
          auto y_it = iomap.find(Y);
          if (y_it == iomap.end()) {
            // Create output tensor if it doesn't exist
            auto y_ptr = x_ptr->copy();
            iomap[Y] = y_ptr;
            y_it = iomap.find(Y);
          } else if (!std::holds_alternative<
                         std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second)) {
            throw std::runtime_error(
                "LeakyReLUNode: Output tensor Y has incorrect type");
          }

          auto y_ptr =
              std::get<std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second);

          TensorOperations<ValueTypeX>::elementwise(
              x_ptr,
              [this](ValueTypeX val) -> ValueTypeX {
                return val < 0 ? alpha * val : val;
              },
              y_ptr);
        }
      },
      x_tensor);
}

std::vector<std::string> LeakyReLUNode::getInputs() { return {X}; }

std::vector<std::string> LeakyReLUNode::getOutputs() { return {Y}; }