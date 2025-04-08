#include "nodes/constant.hpp"
#include "backend/parser_helper.hpp"

ConstantNode::ConstantNode(std::string output, GeneralDataTypes value)
    : output(output), value(value) {}

ConstantNode::ConstantNode(const json &node) {
    if (node.contains("output") && node["output"].is_array()) {
        output = node["output"][0];
    }
    
    if (node.contains("attribute") && node["attribute"].is_array()) {
        for (const auto& attr : node["attribute"]) {
          if (attr["name"] == "value") {
            if (attr["t"].is_object()) {
                // Handle tensor value
                auto t = attr["t"];
                
                int dataType = t["dataType"];

                // Need to handle more data types
                switch (dataType) {
                    case 1:  // FLOAT
                        value = ParserHelper::handle_tensor<float>(t);
                        break;
                    case 2:  // UINT8
                        value = ParserHelper::handle_tensor<uint8_t>(t);
                        break;
                    case 3:  // INT8
                        value = ParserHelper::handle_tensor<int8_t>(t);
                        break;
                    case 4:  // UINT16
                        value = ParserHelper::handle_tensor<uint16_t>(t);
                        break;
                    case 5:  // INT16
                        value = ParserHelper::handle_tensor<int16_t>(t);
                        break;
                    case 6:  // INT32
                        value = ParserHelper::handle_tensor<int32_t>(t);
                        break;
                    case 7:  // INT64
                        value = ParserHelper::handle_tensor<int64_t>(t);
                        break;
                    case 9:  // BOOL
                        value = ParserHelper::handle_tensor<bool>(t);
                        break;
                    case 11: // DOUBLE
                        value = ParserHelper::handle_tensor<double>(t);
                        break;
                    case 12: // UINT32
                        value = ParserHelper::handle_tensor<uint32_t>(t);
                        break;
                    case 13: // UINT64
                        value = ParserHelper::handle_tensor<uint64_t>(t);
                        break;
                    default:
                        throw std::runtime_error("Currently unsupported data type: " + std::to_string(dataType));
                }
            } else {
                throw std::runtime_error("Unsupported value type for Constant node");
            }
          }
        }
    }
}

void ConstantNode::forward(std::unordered_map<std::string, GeneralDataTypes>& iomap) {
    iomap[output] = value;
}

std::vector<std::string> ConstantNode::getInputs() {
    return {};
}

std::vector<std::string> ConstantNode::getOutputs() {
    return {output};
}