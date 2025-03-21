#include "include/flatten_node.hpp"

FlattenNode::FlattenNode(std::string X,
                            std::string Y,
                            int axis) : X(X), Y(Y), axis(axis) {}

FlattenNode::FlattenNode(const json& node) {
    if (node.contains("inputs") && node["inputs"].is_array()) {
        X = node["inputs"][0];
    }

    if (node.contains("outputs") && node["outputs"].is_array()) {
        Y = node["outputs"][0];
    }

    axis = 1;
    if (node.contains("attributes") && node["attributes"].is_object()) {
        for (const auto& attr : node["attributes"]) {
            if (attr["name"] == "axis") {
                axis = attr["i"];
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

        using TensorPtr = std::decay_t<decltype(x_ptr)>;
        using TensorType = typename TensorPtr::element_type;
        using ValueType = typename TensorType::value_type;
        
        if constexpr (!is_in_variant_v<ValueType, T>) {
            throw std::runtime_error("FlattenNode: Unsupported data type for tensor X");
        } else {
            auto y_it = iomap.find(Y);
            if (y_it == iomap.end()) {
                throw std::runtime_error("FlattenNode: Output tensor Y not found in iomap");
            }

            auto y_ptr = std::get<std::shared_ptr<Tensor<ValueType>>>(y_it->second);

            auto input_copy = x_ptr->copy();
            
            if (axis >= input_copy->get_shape().size()) {
                throw std::invalid_argument("Flatten axis is out of range");
            }

            int height_2d, width_2d;

            if (get_axis() == 0) {
                // This gives a warning, but when get_size() returns int in the future it will disappear
                input_copy->reshape({static_cast<int>(input_copy->get_size())});
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

int FlattenNode::get_axis() const {
    return axis;
}