#include "nodes/lrn.hpp"

LRNNode_mml::LRNNode_mml(std::string X, std::string Y, size_t size, float alpha,
                         float beta, float bias)
    : X(X), Y(Y), size(size), alpha(alpha), beta(beta), bias(bias) {};

LRNNode_mml::LRNNode_mml(const nlohmann::json &node) {
  if (node.contains("input") && node["input"].is_array()) {
    X = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }

  size = 1;
  alpha = 0.0001f;
  beta = 0.75f;
  bias = 1.0f;
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto &attr : node["attribute"]) {
      if (attr["name"] == "size") {
        size = std::stoul(attr["i"].get<std::string>());
      } else if (attr["name"] == "alpha") {
        alpha = attr["f"];
      } else if (attr["name"] == "beta") {
        beta = attr["f"];
      } else if (attr["name"] == "bias") {
        bias = attr["f"];
      }
    }
  }
}

void LRNNode_mml::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
    throw std::runtime_error("LRNNode_mml: Input tensor X not found in iomap");
  }

  const GeneralDataTypes &x_tensor = x_it->second;

  std::visit(
      [&](const auto &x_ptr) {
        using ValueTypeX =
            typename std::decay_t<decltype(x_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeX, T>) {
          throw std::runtime_error(
              "LRNNode_mml: Unsupported data type for tensor X");
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
                "LRNNode_mml: Output tensor Y has incorrect type");
          }

          auto y_ptr =
              std::get<std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second);

          array_mml<size_t> shape = x_ptr->get_shape();

          /// Each batch element
          for (size_t n = 0; n < shape[0]; n++) {
            /// Each channel
            for (size_t c = 0; c < shape[1]; c++) {
              /// Each row
              for (size_t h = 0; h < shape[2]; h++) {
                /// Each column
                for (size_t w = 0; w < shape[3]; w++) {

                  /// Region
                  size_t start = std::max(0UL, c - (size - 1) / 2);
                  size_t end = std::min(shape[1] - 1,
                                        c + (size - 1) / 2 + ((size - 1) % 2));

                  /// Calculate square_sum
                  ValueTypeX square_sum = 0;
                  for (size_t i = start; i <= end; i++) {
                    square_sum +=
                        (*x_ptr)[{n, i, h, w}] * (*x_ptr)[{n, i, h, w}];
                  }
                  (*y_ptr)[{n, c, h, w}] =
                      (*x_ptr)[{n, c, h, w}] /
                      std::pow((bias + alpha / size * square_sum), beta);
                }
              }
            }
          }
        }
      },
      x_tensor);
};

std::vector<std::string> LRNNode_mml::getInputs() { return {X}; }

std::vector<std::string> LRNNode_mml::getOutputs() { return {Y}; }