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

MaxPoolNode::MaxPoolNode(const nlohmann::json& node) {
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
            array_mml<size_t> x_shape = x_ptr->get_shape();
            size_t total_rank = x_shape.size();

            if (total_rank < 3) {
                throw std::runtime_error("MaxPoolNode: Input tensor must be at least NCL");
            }

            NodeUtils::compute_pool_attributes(auto_pad, kernel_shape, strides, pads, dilations);

            array_mml<size_t> output_shape = NodeUtils::compute_pool_output_shape(x_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto pad_pair = NodeUtils::compute_pool_pad_begin_end(x_shape, auto_pad, ceil_mode, dilations, kernel_shape, pads, strides);

            auto y_ptr = TensorFactory::create_tensor<ValueType>(output_shape);

            std::optional<std::shared_ptr<Tensor<int64_t>>> indices_ptr = std::nullopt;
            if (indices.has_value()) {
                indices_ptr = TensorFactory::create_tensor<int64_t>(output_shape);
            }

            // Perform pooling operation
            TensorOperationsModule::sliding_window<ValueType>(
                x_shape,
                output_shape,
                kernel_shape,
                strides,
                dilations,
                pad_pair,
                [this, x_ptr, y_ptr, indices_ptr, x_shape, total_rank](const std::vector<std::vector<size_t>>& window_in_idx, const std::vector<size_t>& out_idx) -> void {
                    if (window_in_idx.empty()) {
                        throw std::runtime_error("MaxPoolNode: Empty window values");
                    }

                    ValueType max_val = std::numeric_limits<ValueType>::lowest();
                    int64_t max_idx = -1;

                    for (const auto& in_idx : window_in_idx) {
                        array_mml<size_t> curr_idx(in_idx);
                        ValueType curr_val = (*x_ptr)[curr_idx];
    
                        if (curr_val > max_val) {
                            max_val = curr_val;

                            // Convert in_idx to flat index
                            int64_t flat_index = 0;
                            if (storage_order == 0) { // Row-major
                                for (size_t i = 0; i < total_rank; ++i) {
                                    int64_t stride = 1;
                                    for (size_t j = i + 1; j < total_rank; ++j) {
                                        stride *= x_shape[j];
                                    }
                                    flat_index += in_idx[i] * stride;
                                }
                            } else { // Column-major
                            int64_t stride = 1;
                                for (size_t i = 0; i < total_rank; ++i) {
                                    flat_index += in_idx[i] * stride;
                                    stride *= x_shape[i];
                                }
                            }
                            max_idx = flat_index;
                        }
                    }

                    array_mml<size_t> out_idx_array(out_idx);
                    (*y_ptr)[out_idx_array] = max_val;
                    if (indices_ptr.has_value()) {
                        (*indices_ptr.value())[out_idx_array] = max_idx;
                    }
                }
            );
            
            iomap[Y] = y_ptr;
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