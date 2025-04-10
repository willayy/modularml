#include "nodes/max_pool.hpp"

MaxPoolNode::MaxPoolNode(std::string X, std::string Y, std::vector<int> kernel_shape, 
                         std::optional<std::string> indices,
                         std::string auto_pad, int ceil_mode, std::vector<int> dilations,
                         std::vector<int> pads,
                         int storage_order, std::vector<int> strides) : 
                         X(X), Y(Y), indices(indices), 
                         auto_pad(auto_pad), ceil_mode(ceil_mode), dilations(dilations),
                         kernel_shape(kernel_shape), pads(pads),
                         storage_order(storage_order), strides(strides) {}

MaxPoolNode::MaxPoolNode(const json& node) {
    if (node.contains("input") && node["input"].is_array()) {
        X = node["input"][0];
    }

    indices = std::nullopt;
    if (node.contains("output") && node["output"].is_array()) {
        Y = node["output"][0];
        if (node["input"].size() > 1) {
            indices = node["input"][1];
        }
    }
    
    auto_pad = "NOTSET";
    ceil_mode = 0;
    storage_order = 0;
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
            } else if (attr["name"] == "storage_order") {
                storage_order = std::stoi(attr["i"].get<std::string>());
            }
        }
    }
}

void MaxPoolNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
    auto x_it = iomap.find(X);
    if (x_it == iomap.end()) {
        throw std::runtime_error("MaxPoolNode: Input tensor X not found in iomap");
    }
    
    const GeneralDataTypes& x_tensor = x_it->second;

    std::visit([&](const auto& x_ptr) {
        using ValueType = std::decay_t<decltype(x_ptr)>::element_type::value_type;
        
        if constexpr (!is_in_variant_v<ValueType, T>) {
            throw std::runtime_error("MaxPoolNode: Unsupported data type for tensor X");
        } else {
            if (x_ptr->get_shape().size() < 3) {
                throw std::runtime_error("MaxPoolNode: Input tensor must be at least NCL");
            }

            NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);

            array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(x_ptr->get_shape(), auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto pad_pair = NodeUtils::compute_pool_pad_begin_end(x_ptr->get_shape(), auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto output_ptr = TensorFactory::create_tensor<ValueType>(output_shape);

            std::optional<std::shared_ptr<Tensor<int64_t>>> indices_ptr = std::nullopt;
            if (indices.has_value()) {
                indices_ptr = TensorFactory::create_tensor<int64_t>(output_shape);
            }

            // Perform pooling operation
            TensorOperationsModule::sliding_window<ValueType>(
                x_ptr,
                output_ptr,
                indices_ptr,
                kernel_shape,
                strides,
                dilations,
                pad_pair,
                storage_order,
                [](const std::vector<ValueType>& window_values, const std::vector<int64_t>& window_indices, int64_t& outIndex) -> ValueType {
                    if (window_values.empty()) {
                        throw std::runtime_error("MaxPoolNode: Empty window values");
                    }
                    ValueType max_val = window_values[0];
                    outIndex = window_indices[0];

                    for (size_t i = 1; i < window_values.size(); ++i) {
                        if (window_values[i] > max_val) {
                            max_val = window_values[i];
                            outIndex = window_indices[i];
                        }
                    }

                    return max_val;
                }
            );
            
            iomap[Y] = output_ptr;
            if (indices.has_value()) {
                iomap[indices.value()] = indices_ptr.value();
            }
        }
    }, x_tensor);
}

std::vector<std::string> MaxPoolNode::getInputs() { return {X}; }

std::vector<std::string> MaxPoolNode::getOutputs() { 
    if (indices.has_value()) {
        return {Y, indices.value()};
    } else {
        return {Y};
    }
}