#include "nodes/flatten.hpp"

FlattenNode::FlattenNode(std::string X,
                            std::string Y,
                            int axis) : X(X), Y(Y), axis(axis) {}

FlattenNode::FlattenNode(const json& node) {
    if (node.contains("input") && node["input"].is_array()) {
        X = node["input"][0];
    }

    if (node.contains("output") && node["output"].is_array()) {
        Y = node["output"][0];
    }

    axis = 1;
    if (node.contains("attribute") && node["attribute"].is_array()) {
        for (const auto& attr : node["attribute"]) {
            if (attr["name"] == "axis") {
                axis = std::stoul(attr["i"].get<std::string>());
            }
        }
    }
}
void FlattenNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
    auto x_it = iomap.find(X);
    if (x_it == iomap.end()) {
        throw std::runtime_error("FlattenNode: Input tensor X not found in iomap");
    }
    
    const GeneralDataTypes& x_tensor = x_it->second;

    std::visit([&](const auto& x_ptr) {
        using ValueType = typename std::decay_t<decltype(x_ptr)>::element_type::value_type;
        
        if constexpr (!is_in_variant_v<ValueType, T>) {
            throw std::runtime_error("FlattenNode: Unsupported data type for tensor X");
        } else {
            auto y_it = iomap.find(Y);
            if (y_it == iomap.end()) {
                // Create output tensor if it doesn't exist
                auto y_ptr = x_ptr->copy();
                // No need to fill with zeros as the flatten function will overwrite the values
                iomap[Y] = y_ptr;
                y_it = iomap.find(Y);
            } else if (!std::holds_alternative<std::shared_ptr<Tensor<ValueType>>>(y_it->second)) {
                throw std::runtime_error("FlattenNode: Output tensor Y has incorrect type");
            }

            auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);

            auto input_copy = x_ptr->copy();
            
            if (axis >= input_copy->get_shape().size()) {
                throw std::invalid_argument("Flatten axis is out of range");
            }

            uli height_2d, width_2d;

            if (get_axis() == 0) {
                // This gives a warning, but when get_size() returns int in the future it will disappear
                input_copy->reshape({input_copy->get_size()});
            } else { 
                height_2d = 1;
                width_2d = 1;
                
                int i = 0;
                for (i; i < axis; i++) {
                    height_2d *= input_copy->get_shape()[i];
                }
                for (i; i < input_copy->get_shape().size(); i++) {
                    width_2d *= input_copy->get_shape()[i];
                }
            }

            input_copy->reshape({height_2d, width_2d});

            *y_ptr = *input_copy;
        }
    }, x_tensor);
}

std::vector<std::string> FlattenNode::getInputs() {
    return {X};
}

std::vector<std::string> FlattenNode::getOutputs() {
    return {Y};
}

int FlattenNode::get_axis() const {
    return axis;
}