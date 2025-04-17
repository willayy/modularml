#include "nodes/gelu.hpp"

#include <algorithm>

// IWYU pragma: no_include <__math/exponential_functions.h>
// IWYU pragma: no_include <__math/hyperbolic_functions.h>
// IWYU pragma: no_include <__math/roots.h>
// IWYU pragma: no_include <__math/error_functions.h>
#include <cmath>  // IWYU pragma: keep
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json.hpp"

GeluNode::GeluNode(std::string X, std::string Y, std::string approximate)
    : X(X), Y(Y) {
  if (approximate == "none" || approximate == "tanh") {
    this->approximate = approximate;
  } else {
    throw std::invalid_argument("Invalid value for argument approximate.");
  }
}

GeluNode::GeluNode(const nlohmann::json &node) {
  if (node.contains("input") && node["input"].is_array()) {
    X = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }

  approximate = "none";
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto &attr : node["attribute"]) {
      if (attr["name"] == "approximate") {
        approximate = attr["s"];
      }
    }
  }
}

void GeluNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
    throw std::runtime_error("GELUNode: Input tensor X not found in iomap");
  }

  const GeneralDataTypes &x_tensor = x_it->second;

  std::visit(
      [&](const auto &x_ptr) {
        using ValueTypeX =
            typename std::decay_t<decltype(x_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeX, T>) {
          throw std::runtime_error(
              "GELUNode: Unsupported data type for tensor X");
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
                "GELUNode: Output tensor Y has incorrect type");
          }

          auto y_ptr =
              std::get<std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second);

          if (approximate == "none") {
            TensorOperations::elementwise<ValueTypeX>(
                x_ptr,
                [](ValueTypeX val) -> ValueTypeX {
                  return static_cast<ValueTypeX>(
                      0.5f * val * (1.0f + std::erf(val / std::sqrt(2.0f))));
                },
                y_ptr);
          } else {
            TensorOperations::elementwise<ValueTypeX>(
                x_ptr,
                [](ValueTypeX val) -> ValueTypeX {
                  return static_cast<ValueTypeX>(
                      0.5f * val *
                      (1.0f +
                       std::tanh(std::sqrt(2.0f / M_PI) *
                                 (val + 0.044715f * std::pow(val, 3.0f)))));
                },
                y_ptr);
          }
        }
      },
      x_tensor);
}

std::vector<std::string> GeluNode::getInputs() { return {X}; }

std::vector<std::string> GeluNode::getOutputs() { return {Y}; }