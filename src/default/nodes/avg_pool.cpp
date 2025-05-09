#include "nodes/avg_pool.hpp"

AvgPoolNode::AvgPoolNode(const std::string& X, const std::string& Y,
                         const std::vector<int>& kernel_shape,
                         const std::string& auto_pad, int ceil_mode,
                         int count_include_pad,
                         const std::vector<int>& dilations,
                         const std::vector<int>& pads,
                         const std::vector<int>& strides)
    : X(X),
      Y(Y),
      auto_pad(auto_pad),
      ceil_mode(ceil_mode),
      count_include_pad(count_include_pad),
      dilations(dilations),
      kernel_shape(kernel_shape),
      pads(pads),
      strides(strides) {}

AvgPoolNode::AvgPoolNode(const nlohmann::json& node) {
  if (node.contains("input") && node["input"].is_array()) {
    X = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    Y = node["output"][0];
  }

  auto_pad = "NOTSET";
  ceil_mode = 0;
  count_include_pad = 0;
  dilations = {};
  pads = {};
  strides = {};
  if (node.contains("attribute") && node["attribute"].is_array()) {
    for (const auto& attr : node["attribute"]) {
      if (attr["name"] == "kernel_shape") {
        std::vector<int> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(std::stoi(val.get<std::string>()));
        }
        kernel_shape = values;
      } else if (attr["name"] == "strides") {
        std::vector<int> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(std::stoi(val.get<std::string>()));
        }
        strides = values;
      } else if (attr["name"] == "auto_pad") {
        auto_pad = attr["s"];
      } else if (attr["name"] == "ceil_mode") {
        ceil_mode = std::stoi(attr["i"].get<std::string>());
      } else if (attr["name"] == "dilations") {
        std::vector<int> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(std::stoi(val.get<std::string>()));
        }
        dilations = values;
      } else if (attr["name"] == "pads") {
        std::vector<int> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(std::stoi(val.get<std::string>()));
        }
        pads = values;
      } else if (attr["name"] == "count_include_pad") {
        count_include_pad = std::stoi(attr["i"].get<std::string>());
      }
    }
  }
}

void AvgPoolNode::forward(
    std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto x_it = iomap.find(X);
  if (x_it == iomap.end()) {
    throw std::runtime_error("AvgPoolNode: Input tensor X not found in iomap");
  }

  const GeneralDataTypes& x_tensor = x_it->second;

  std::visit(
      [&](const auto& x_ptr) {
        using ValueType =
            std::decay_t<decltype(x_ptr)>::element_type::value_type;

        if constexpr (!is_in_variant_v<ValueType, T>) {
          throw std::runtime_error(
              "AvgPoolNode: Unsupported data type for tensor X");
        } else {
          array_mml<size_t> x_shape = x_ptr->get_shape();
          size_t total_rank = x_shape.size();

          if (total_rank < 3) {
            throw std::runtime_error(
                "AvgPoolNode: Input tensor must be at least NCL");
          }

          NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides,
                                             pads, dilations);

          array_mml<size_t> output_shape = NodeUtils::compute_pool_output_shape(
              x_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads,
              strides);

          auto pad_pair = NodeUtils::compute_pool_pad_begin_end(
              x_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads,
              strides);

          auto y_ptr = std::make_shared<Tensor<ValueType>>(output_shape);

          // Perform pooling operation
          TensorOperations<ValueType>::sliding_window(
              x_shape, output_shape, kernel_shape, strides, dilations, pad_pair,
              [this, x_ptr, y_ptr](
                  const std::vector<std::vector<size_t>>& window_in_idx,
                  const std::vector<size_t>& out_idx) -> void {
                if (window_in_idx.empty()) {
                  throw std::runtime_error("AvgPoolNode: Empty window values");
                }

                ValueType sum = 0;
                for (const auto& in_idx : window_in_idx) {
                  array_mml<size_t> curr_idx(in_idx);
                  sum += (*x_ptr)[curr_idx];
                }

                int kernel_volume = 1;
                for (auto k : kernel_shape) {
                  kernel_volume *= k;
                }

                int denominator = count_include_pad
                                      ? kernel_volume
                                      : static_cast<int>(window_in_idx.size());

                array_mml<size_t> out_idx_array(out_idx);
                (*y_ptr)[out_idx_array] =
                    sum / static_cast<ValueType>(denominator);
              });

          iomap[Y] = y_ptr;
        }
      },
      x_tensor);
}

std::vector<std::string> AvgPoolNode::getInputs() { return {X}; }

std::vector<std::string> AvgPoolNode::getOutputs() { return {Y}; }