#include "include/reshape_node.hpp"

reshapeNode::reshapeNode(std::string data, std::string shape,
                            std::string reshaped, int allowzero)
    : data(data), shape(shape), reshaped(reshaped), allowzero(allowzero) {
  if (allowzero != 0 && allowzero != 1)
    throw runtime_error("Invalid value for allowzero. Must be 0 or 1.");
}

reshapeNode::reshapeNode(const json& node) {
  if (node.contains("inputs") && node["inputs"].is_array()) {
    data = node["inputs"][0];
    shape = node["inputs"][1];
  }

  if (node.contains("outputs") && node["outputs"].is_array()) {
    reshaped = node["outputs"][0];
  }

  allowzero = 0;
  if (node.contains("attributes") && node["attributes"].is_object()) {
    for (const auto& attr : node["attributes"]) {
      if (attr["name"] == "allowzero") {
        allowzero = attr["i"];
      }
    }
  }
}

void reshapeNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
  auto data_it = iomap.find(data);
  if (data_it == iomap.end()) {
      throw std::runtime_error("ReshapeNode: Input tensor data not found in iomap");
  }

  auto shape_it = iomap.find(shape);
  if (shape_it == iomap.end()) {
      throw std::runtime_error("ReshapeNode: Input tensor shape not found in iomap");
  }

  const GeneralDataTypes& data_tensor = data_it->second;
  const GeneralDataTypes& shape_tensor = shape_it->second;

  std::visit([&](const auto& data_ptr, const auto& shape_ptr) {
    using ValueType = typename std::decay_t<decltype(data_ptr)>::element_type::value_type;
    using ShapeValueType = typename std::decay_t<decltype(shape_ptr)>::element_type::value_type;

    if constexpr (!is_in_variant_v<ValueType, T> || !is_in_variant_v<ShapeValueType, ShapeDataType>) {
      throw std::runtime_error("ReshapeNode: Unsupported data type for tensor data");
    } else {
      auto reshaped_it = iomap.find(reshaped);
      if (reshaped_it == iomap.end()) {
        throw std::runtime_error("ReshapeNode: Output tensor reshaped not found in iomap");
      }

      auto reshaped_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(reshaped_it->second);

      // Determine the size of the shape tensor (number of dimensions for the new shape)
      int shape_size = shape_ptr->get_size();

      // Determine the total number of elements in the input data tensor
      int data_size = data_ptr->get_size();

      // Create an array to store the new shape values (initialized with same size as shape tensor)
      array_mml<int> new_shape(shape_size);

      // Variables for handling inferred dimension (-1) and computing the total number of elements
      int inferred_dim_index = -1;  // Stores the index of -1 if present
      int computed_elements = 1;    // Tracks the product of explicitly defined shape dimensions
      
      // Iterate through the shape tensor to determine the new shape values
      for (int i = 0; i < shape_size; ++i) {
        int dim = (*shape_ptr)[i];  // Extract the value for the current dimension

        // If dim == -1, mark it for inference (meaning this dimension should be computed)
        if (dim == -1) {
          // Ensure that only one dimension is set to -1 (otherwise, the reshape would be ambiguous)
          if (inferred_dim_index != -1) {
            throw runtime_error("Invalid reshape: multiple -1 values in shape tensor.");
          }
          inferred_dim_index = i;  // Store the index of the inferred dimension
          new_shape[i] = -1;       // Placeholder for now (to be computed later)

          // If dim == 0 and allowzero flag is set, keep the original input shape at this index
        } else if (dim == 0 && allowzero == 1) {
          new_shape[i] = data_ptr->get_shape()[i];  // Copy corresponding dimension from input tensor
          computed_elements *= new_shape[i];    // Update total number of elements
        } else {
          // Otherwise, just assign the given dimension value
          new_shape[i] = dim;
          computed_elements *= dim;
        }
      }

      // If -1 was found, infer its value to ensure the total number of elements remains correct
      if (inferred_dim_index != -1) {
        // Ensure that the total number of elements in the new shape matches the original data size
        if (data_size % computed_elements != 0) {
          throw runtime_error("Invalid reshape: inferred dimension does not match total elements.");
        }
        // Compute the missing dimension size and update the new shape
        new_shape[inferred_dim_index] = data_size / computed_elements;

        // Update total computed elements (not strictly necessary, but keeps logic consistent)
        computed_elements *= new_shape[inferred_dim_index];
      }

      // Copy the data from the input tensor to the reshaped output tensor
      *reshaped_ptr = *data_ptr;

      // Apply reshape with the calculated new shape
      reshaped_ptr->reshape(new_shape);
    }
  }, data_tensor, shape_tensor);
}

std::vector<std::string> reshapeNode::getInputs() {
  return {data, shape};
}

std::vector<std::string> reshapeNode::getOutputs() {
  return {reshaped};
}