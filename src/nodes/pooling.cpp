#include "nodes/pooling.hpp"

PoolingNode_mml::PoolingNode_mml(std::string input, std::vector<std::string> outputs, 
                                array_mml<uli> kernel_shape, array_mml<uli> strides,
                                string auto_pad, uli ceil_mode,
                                array_mml<uli> dilations,
                                array_mml<uli> pads)
    : input(input), outputs(outputs), kernel_shape(kernel_shape), strides(strides), 
      dilations(dilations), pads(pads) {
  if (auto_pad != "NOTSET" && auto_pad != "VALID" && auto_pad != "SAME_UPPER" &&
      auto_pad != "SAME_LOWER") {
    throw std::invalid_argument("Invalid padding value! Only 'VALID', "
                                "'SAME_UPPER' and 'SAME_LOWER' are allowed.");
  }
  if (ceil_mode < 0 || ceil_mode > 1) {
    throw std::invalid_argument("Invalid ceil_mode value! Must be 0 or 1");
  }

  this->auto_pad = auto_pad;
  this->ceil_mode = ceil_mode;
};

PoolingNode_mml::PoolingNode_mml(const json& node) {
  if (node.contains("input") && node["input"].is_array()) {
    input = node["input"][0];
  }

  if (node.contains("output") && node["output"].is_array()) {
    outputs = node["output"];
  }

  kernel_shape = {1, 1};
  strides = {1, 1};
  auto_pad = "NOTSET";
  ceil_mode = 0;
  dilations = {1, 1};
  pads = {0, 0, 0, 0};
  if (node.contains("attribute") && node["attribute"].is_object()) {
    for (const auto& attr : node["attribute"]) {
      if (attr["name"] == "kernel_shape") {
        std::vector<uli> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(static_cast<uli>(std::stoul(val.get<std::string>())));
        }
        kernel_shape = array_mml<uli>(values);
      } else if (attr["name"] == "strides") {
        std::vector<uli> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(static_cast<uli>(std::stoul(val.get<std::string>())));
        }
        strides = array_mml<uli>(values);
      } else if (attr["name"] == "auto_pad") {
        auto_pad = attr["s"];
      } else if (attr["name"] == "ceil_mode") {
        ceil_mode = attr["i"];
      } else if (attr["name"] == "dilations") {
        std::vector<uli> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(static_cast<uli>(std::stoul(val.get<std::string>())));
        }
        dilations = array_mml<uli>(values);
      } else if (attr["name"] == "pads") {
        std::vector<uli> values;
        for (const auto& val : attr["ints"]) {
          values.push_back(static_cast<uli>(std::stoul(val.get<std::string>())));
        }
        pads = array_mml<uli>(values);
      }
    }
  }
}

void PoolingNode_mml::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto input_it = iomap.find(input);
  if (input_it == iomap.end()) {
    throw std::runtime_error("PoolingNode_mml: Input tensor not found in iomap");
  }

  const GeneralDataTypes& input = input_it->second;

  std::visit([&](const auto& input_ptr) {
    using ValueType = typename std::decay_t<decltype(input_ptr)>::element_type::value_type;

    if constexpr (!is_in_variant_v<ValueType, T>) {
      throw std::runtime_error("PoolingNode_mml: Unsupported data type for tensor X");
    } else {
      array_mml<uli> input_shape = input_ptr->get_shape();
      if (input_shape.size() != 4) {
        throw std::invalid_argument("Invalid tensor shape");
      }

      array_mml<uli> output_shape = array_mml({input_shape[0], input_shape[1], 1UL, 1UL});

      // Calculate effective kernel size with dilation
      array_mml<uli> effective_kernel_shape = 
        array_mml({kernel_shape[0] + (kernel_shape[0] - 1) * (dilations[0] - 1), 
                   kernel_shape[1] + (kernel_shape[1] - 1) * (dilations[1] - 1)});
          
      vector<uli> pad_shape = {pads[0] + pads[1], pads[2] + pads[3]};

      // Calculate output dimensions based on padding type
      for (uli i = 2; i < 4; i++) {
        if (auto_pad == "VALID") {
          if (ceil_mode) {
            output_shape[i] = static_cast<uli>(
                ceil((static_cast<float>(input_shape[i]) -
                      (effective_kernel_shape[i - 2] - 1) * dilations[i - 2]) /
                    static_cast<float>(strides[i - 2])));

          } else {

            output_shape[i] =
                (input_shape[i] -
                (effective_kernel_shape[i - 2] - 1) * dilations[i - 2]) /
                    strides[i - 2] +
                1;
          }
        } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {

          if (ceil_mode) {

            output_shape[i] = static_cast<uli>(
                ceil(static_cast<float>(input_shape[i]) / strides[i - 2]));

          } else {
            output_shape[i] =
                static_cast<uli>(floor((static_cast<float>(input_shape[i]) - 1) /
                                      static_cast<float>(strides[i - 2]))) +
                1;
          }
          pad_shape[i - 2] =
              (output_shape[i] - 1) * strides[i - 2] +
              ((effective_kernel_shape[i - 2] - 1) * dilations[i - 2] + 1) -
              input_shape[i];

        } else {

          if (ceil_mode) {
            output_shape[i] = static_cast<uli>(
                ceil((static_cast<float>(input_shape[i]) + pad_shape[i - 2] -
                      dilations[i - 2] * (effective_kernel_shape[i - 2] - 1) - 1) /
                        strides[i - 2] +
                    1));
          } else {

            output_shape[i] = static_cast<uli>(
                floor((input_shape[i] + pad_shape[i - 2] -
                      dilations[i - 2] * (effective_kernel_shape[i - 2] - 1) - 1) /
                          strides[i - 2] +
                      1));
          }
        }
      }

      pooling(input_ptr, input_shape, output_shape, effective_kernel_shape,
        pad_shape[0], pad_shape[1], auto_pad, iomap);
    }
  }, input);
}

std::vector<std::string> PoolingNode_mml::getInputs() { return {input}; }

std::vector<std::string> PoolingNode_mml::getOutputs() { return outputs; }