#pragma once

#include "datastructures/a_tensor.hpp"
#include "../utility/uli.hpp"
#include "utility/base64.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace ParserHelper {
/**
 * @brief Helper std::function to create a tensor from JSON data.
 *
 * @tparam T The type of the tensor elements.
 * @param init The JSON object containing the tensor data.
 * @param shapeArray The shape of the tensor.
 * @return A shared pointer to the created tensor.
 */
template <typename T>
std::shared_ptr<Tensor<T>> handle_tensor(const nlohmann::json &init) {
  std::vector<uli> dims;
  for (const auto &el : init["dims"]) {
    dims.push_back(static_cast<uli>(std::stoi(el.get<std::string>())));
  }
  array_mml shapeArray(dims);

  if (init.contains("rawData")) {
    return std::make_shared<Tensor_mml<T>>(
        shapeArray, Base64::decode<T>(init["rawData"].get<std::string>()));
  } else {
    std::string fieldName;

    if constexpr (std::is_same_v<T, float>)
      fieldName = "floatData";
    else if constexpr (std::is_same_v<T, double>)
      fieldName = "doubleData";
    else if constexpr (std::is_same_v<T, int64_t>)
      fieldName = "int64Data";
    else if constexpr (std::is_same_v<T, int32_t>)
      fieldName = "int32Data";
    else if constexpr (std::is_same_v<T, uint64_t>)
      fieldName = "uint64Data";
    else if constexpr (std::is_same_v<T, uint32_t>)
      fieldName = "uint32Data";
    else if constexpr (std::is_same_v<T, uint16_t>)
      fieldName = "uint16Data";
    else if constexpr (std::is_same_v<T, int16_t>)
      fieldName = "int16Data";
    else if constexpr (std::is_same_v<T, uint8_t>)
      fieldName = "uint8Data";
    else if constexpr (std::is_same_v<T, int8_t>)
      fieldName = "int8Data";
    else if constexpr (std::is_same_v<T, bool>)
      fieldName = "boolData";
    else
      fieldName = "unknownData";

    if (init.contains(fieldName)) {
      std::vector<T> data;
      for (const auto &el : init[fieldName]) {
        if (el.is_number()) {
          data.push_back(el.get<T>());
        } else if (el.is_string()) {
          if constexpr (std::is_same_v<T, bool>) {
            // Handle boolean strings like "true", "false", "1", "0"
            std::string value = el.get<std::string>();
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            if (value == "true" || value == "1" || value == "yes") {
              data.push_back(true);
            } else if (value == "false" || value == "0" || value == "no") {
              data.push_back(false);
            } else {
              throw std::runtime_error("Invalid boolean std::string: " + value);
            }
          } else if constexpr (std::is_same_v<T, int64_t>) {
            data.push_back(static_cast<T>(std::stoll(el.get<std::string>())));
          } else if constexpr (std::is_same_v<T, uint64_t>) {
            data.push_back(static_cast<T>(std::stoull(el.get<std::string>())));
          } else if constexpr (std::is_integral_v<T>) {
            data.push_back(static_cast<T>(std::stoi(el.get<std::string>())));
          } else if constexpr (std::is_floating_point_v<T>) {
            data.push_back(static_cast<T>(std::stod(el.get<std::string>())));
          } else {
            throw std::runtime_error(
                "Invalid std::string conversion for type: " +
                init["name"].get<std::string>());
          }
        } else if (el.is_boolean()) {
          data.push_back(static_cast<T>(el.get<bool>()));
        } else {
          throw std::runtime_error("Invalid data type in tensor: " +
                                   init["name"].get<std::string>());
        }
      }
      array_mml<T> dataArray(data);
      return std::make_shared<Tensor_mml<T>>(shapeArray, dataArray);
    } else {
      throw std::runtime_error("No data field found for tensor: " +
                               init["name"].get<std::string>());
    }
  }
}
} // namespace ParserHelper