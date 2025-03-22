#include "include/Dropout_node.hpp"

DropoutNode::DropoutNode(std::string data,
                            std::string output,
                            optional<std::string> mask,
                            float ratio,
                            bool training_mode,
                            optional<int> seed)
    : data(data), output(output), mask(mask), ratio(ratio), training_mode(training_mode), seed(seed) {}

DropoutNode::DropoutNode(const json& node) {
  if (node.contains("inputs") && node["inputs"].is_array()) {
    data = node["inputs"][0];
  }

  if (node.contains("outputs") && node["outputs"].is_array()) {
    output = node["outputs"][0];
    if (node["outputs"].size() > 1) {
      mask = node["outputs"][1];
    }
  }

  ratio = 0.5;
  training_mode = false;
  seed = std::nullopt;
  if (node.contains("attributes") && node["attributes"].is_object()) {
    for (const auto& attr : node["attributes"]) {
      if (attr["name"] == "ratio") {
        ratio = attr["f"];
      } else if (attr["name"] == "training_mode") {
        training_mode = attr["i"];
      } else if (attr["name"] == "seed") {
        seed = attr["i"];
      }
    }
  }
}

void DropoutNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto data_it = iomap.find(data);
  if (data_it == iomap.end()) {
      throw std::runtime_error("ReshapeNode: Input tensor data not found in iomap");
  }

  const GeneralDataTypes& data_tensor = data_it->second;

  std::visit([&](const auto& data_ptr) {
    using TensorPtr = std::decay_t<decltype(data_ptr)>;
    using TensorType = typename TensorPtr::element_type;
    using ValueType = typename TensorType::value_type;

    if constexpr (!is_in_variant_v<ValueType, T>) {
      throw std::runtime_error("ReshapeNode: Unsupported data type for tensor data");
    } else {
      auto output_it = iomap.find(output);
      if (output_it == iomap.end()) {
        throw std::runtime_error("ReshapeNode: Output tensor reshaped not found in iomap");
      }

      auto output_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(output_it->second);

      if (data_ptr->get_shape().size() < 1) {
        throw runtime_error("Tensor data must be at least 1D.");
      }

      if (training_mode) {
        throw runtime_error("DropoutNode forward pass in training mode is not implemented yet.");
      } else {
        *output_ptr = *data_ptr;
      }
    }
  }, data_tensor);
}

std::vector<std::string> DropoutNode::getInputs() {
  return {data};
}

std::vector<std::string> DropoutNode::getOutputs() {
  if (mask.has_value()) {
    return {output, mask.value()};
  } else {
    return {output};
  }
}