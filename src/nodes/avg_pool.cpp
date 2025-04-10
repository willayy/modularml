#include "nodes/avg_pool.hpp"

AvgPoolNode::AvgPoolNode(std::string X, std::string Y, std::vector<int> kernel_shape,
                         std::string auto_pad, int ceil_mode, int count_include_pad, std::vector<int> dilations,
                         std::vector<int> pads, std::vector<int> strides) : 
                         X(X), Y(Y), auto_pad(auto_pad), ceil_mode(ceil_mode), 
                         count_include_pad(count_include_pad), dilations(dilations), 
                         kernel_shape(kernel_shape), pads(pads), strides(strides) {}

AvgPoolNode::AvgPoolNode(const json& node) {
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

void AvgPoolNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
    auto x_it = iomap.find(X);
    if (x_it == iomap.end()) {
        throw std::runtime_error("AvgPoolNode: Input tensor X not found in iomap");
    }
    
    const GeneralDataTypes& x_tensor = x_it->second;

    std::visit([&](const auto& x_ptr) {
        using ValueType = std::decay_t<decltype(x_ptr)>::element_type::value_type;
        
        if constexpr (!is_in_variant_v<ValueType, T>) {
            throw std::runtime_error("AvgPoolNode: Unsupported data type for tensor X");
        } else {
            if (x_ptr->get_shape().size() < 3) {
                throw std::runtime_error("AvgPoolNode: Input tensor must be at least NCL");
            }

            NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);

            array_mml<uli> output_shape = NodeUtils::compute_pool_output_shape(x_ptr->get_shape(), auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto pad_pair = NodeUtils::compute_pool_pad_begin_end(x_ptr->get_shape(), auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto output_ptr = TensorFactory::create_tensor<ValueType>(output_shape);

            // Perform pooling operation
            TensorOperationsModule::sliding_window<ValueType>(
                x_ptr,
                output_ptr,
                std::nullopt,
                kernel_shape,
                strides,
                dilations,
                pad_pair,
                0,
                [this](const std::vector<ValueType>& window_values, const std::vector<int64_t>& window_indices, int64_t& outIndex) -> ValueType {
                    if (window_values.empty()) {
                        throw std::runtime_error("AvgPoolNode: Empty window values");
                    }

                    ValueType sum = 0;
                    for (const auto& val : window_values) {
                        sum += val;
                    }

                    int kernel_volume = 1;
                    for (auto k : kernel_shape) {
                        kernel_volume *= k;
                    }

                    int denominator = count_include_pad ? kernel_volume : static_cast<int>(window_values.size());

                    return sum / static_cast<ValueType>(denominator);
                }
            );
            
            iomap[Y] = output_ptr;
        }
    }, x_tensor);
}

std::vector<std::string> AvgPoolNode::getInputs() { return {X}; }

std::vector<std::string> AvgPoolNode::getOutputs() { return {Y}; }