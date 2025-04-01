#pragma once

#include "a_data_parser.hpp"
#include "utility/base64.hpp"
/**
 * @class Parser_mml
 * @brief A class for parsing JSON data of a model into a Model_mml object.
 */
class Parser_mml: public DataParser {
  public:
    /**
     * @brief Default constructor for Parser_mml.
     */
    Parser_mml() = default;
    
    /**
     * @brief Parses JSON data of a model into a Model_mml object.
     *
     * @param data JSON data of a Model.
     * @return The default representation of a model: Model_mml.
     */
    unique_ptr<Model> parse(const json& data) const override;
};

template <typename T>
void handleTensorData(const json& init, const std::string& initName, 
                     const array_mml<uli>& shapeArray, 
                     std::unordered_map<std::string, GeneralDataTypes>& tensorMap) {
  if (init.contains("rawData")) {
    array_mml<T> dataArray = base64::decode<T>(init["rawData"].get<std::string>());
    tensorMap[initName] = std::make_shared<Tensor_mml<T>>(shapeArray, dataArray);
  } else {
    std::string fieldName;

    if constexpr (std::is_same_v<T, float>) fieldName = "floatData";
    else if constexpr (std::is_same_v<T, double>) fieldName = "doubleData";
    else if constexpr (std::is_same_v<T, int64_t>) fieldName = "int64Data";
    else if constexpr (std::is_same_v<T, int32_t>) fieldName = "int32Data";
    else if constexpr (std::is_same_v<T, uint64_t>) fieldName = "uint64Data";
    else if constexpr (std::is_same_v<T, uint32_t>) fieldName = "uint32Data";
    else if constexpr (std::is_same_v<T, uint16_t>) fieldName = "uint16Data";
    else if constexpr (std::is_same_v<T, int16_t>) fieldName = "int16Data";
    else if constexpr (std::is_same_v<T, uint8_t>) fieldName = "uint8Data";
    else if constexpr (std::is_same_v<T, int8_t>) fieldName = "int8Data";
    else if constexpr (std::is_same_v<T, bool>) fieldName = "boolData";
    else fieldName = "unknownData";

    if (init.contains(fieldName)) {
      std::vector<T> data;
      for (const auto& el : init[fieldName]) {
        if (el.is_number()) {
          data.push_back(el.get<T>());
        } else if (el.is_string()) {
          if constexpr (std::is_same_v<T, bool>) {
            // Handle boolean strings like "true", "false", "1", "0"
            std::string value = el.get<std::string>();
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
                           
            if (value == "true" || value == "1" || value == "yes") {
              data.push_back(true);
            } else if (value == "false" || value == "0" || value == "no") {
              data.push_back(false);
            } else {
              throw std::runtime_error("Invalid boolean string: " + value);
            }
          } 
          else if constexpr (std::is_same_v<T, int64_t>) {
            data.push_back(static_cast<T>(std::stoll(el.get<std::string>())));
          }
          else if constexpr (std::is_same_v<T, uint64_t>) {
            data.push_back(static_cast<T>(std::stoull(el.get<std::string>())));
          }
          else if constexpr (std::is_integral_v<T>) {
            data.push_back(static_cast<T>(std::stoi(el.get<std::string>())));
          }
          else if constexpr (std::is_floating_point_v<T>) {
            data.push_back(static_cast<T>(std::stod(el.get<std::string>())));
          }
          else {
            throw std::runtime_error("Invalid string conversion for type: " + initName);
          }
        } else if (el.is_boolean()) {
          data.push_back(static_cast<T>(el.get<bool>()));
        } else {
          throw std::runtime_error("Invalid data type in tensor: " + initName);
        }
      }
      array_mml<T> dataArray(data);
      tensorMap[initName] = std::make_shared<Tensor_mml<T>>(shapeArray, dataArray);
    } else {
      throw std::runtime_error("No data field found for tensor: " + initName);
    }
  }
}
