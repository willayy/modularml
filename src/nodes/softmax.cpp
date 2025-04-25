#include "nodes/softmax.hpp"

#include <algorithm>
// IWYU pragma: no_include <__math/exponential_functions.h>
// IWYU pragma: no_include <__math/logarithms.h>
#include <cmath>  // IWYU pragma: keep
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
// IWYU pragma: no_include <__vector/vector.h>
#include <vector>  // IWYU pragma: keep

#include "nlohmann/json.hpp"

SoftMaxNode::SoftMaxNode(const std::string &X, const std::string &Y, size_t axis)
    : X(X), Y(Y), axis(axis) {}

SoftMaxNode::SoftMaxNode(const nlohmann::json &node) {
  if (node.contains("input") && node["input"].is_array()) {
    X = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }

  axis = -1;
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto &attr : node["attribute"]) {
      if (attr["name"] == "axis") {
        axis = std::stoul(attr["i"].get<std::string>());
      }
    }
  }
}

void SoftMaxNode::forward(
    std::unordered_map<std::string, GeneralDataTypes> &iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
    throw std::runtime_error(
        "SoftMaxNode: Input tensor X not found in iomap");
  }

  const GeneralDataTypes &x_tensor = x_it->second;

  std::visit(
      [&](const auto &x_ptr) {
        using ValueTypeX =
            typename std::decay_t<decltype(x_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueTypeX, T>) {
          throw std::runtime_error(
              "SoftMaxNode: Unsupported data type for tensor X");
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
                "SoftMaxNode: Output tensor Y has incorrect type");
          }

          auto y_ptr =
              std::get<std::shared_ptr<Tensor<ValueTypeX>>>(y_it->second);

          // If axis is negative
          if (((int)axis) < 0) axis += x_ptr->get_shape().size();

          if (axis >= x_ptr->get_shape().size())
            throw std::runtime_error("Invalid axis: " + std::to_string(axis));

          // Currently this only supports input tensors that are 2D, this is the
          // most common shape. Currently there is no general solution until we
          // can slice the tensor and retreive axi from the tensor

          auto input_copy = x_ptr->copy();

          // For each batch, in the input
          for (size_t b = 0; b < input_copy->get_shape()[0]; b++) {
            // Find the maximum value in the row for numerical stability
            ValueTypeX max_val = -std::numeric_limits<ValueTypeX>::infinity();
            for (size_t c = 0; c < input_copy->get_shape()[axis]; c++) {
              max_val = std::max(max_val, (*input_copy)[{b, c}]);
            }

            // Exponentiate and std::accumulate the sum
            ValueTypeX sum = 0;
            std::vector<ValueTypeX> exp_values(input_copy->get_shape()[axis]);
            for (size_t c = 0; c < input_copy->get_shape()[axis]; c++) {
              ValueTypeX value = (*input_copy)[{b, c}] - max_val;
              exp_values[c] = std::exp(value);
              sum += exp_values[c];
            }

            for (size_t c = 0; c < input_copy->get_shape()[axis]; c++) {
              // Apply soft max and perform std::log on the result
              (*input_copy)[{b, c}] = exp_values[c] / sum;
            }
          }

          *y_ptr = *input_copy;
        }
      },
      x_tensor);
}

std::vector<std::string> SoftMaxNode::getInputs() { return {X}; }

std::vector<std::string> SoftMaxNode::getOutputs() { return {Y}; }