#pragma once

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
#include <vector>  // IWYU pragma: keep

#include "model/a_model.hpp"

/**
 * @class DataParser
 * @brief Abstract base class for parsing model definitions from JSON format.
 *
 * This class defines the interface for parsers that convert model definitions
 * from a serialized JSON format into executable Model objects. Different parser
 * implementations can support various model definition formats (e.g., ONNX,
 * TensorFlow, custom formats) by extending this class.
 */
class DataParser {
 public:
  /**
   * @brief Parses a JSON model definition into an executable Model object.
   *
   * This pure virtual function must be implemented by derived classes to define
   * how specific model definition formats are parsed and converted into Model
   * objects. The implementation should handle all aspects of model
   * construction, including creating the appropriate nodes and establishing
   * their connections.
   *
   * @param data JSON data containing the model definition
   * @return A unique pointer to the constructed Model object
   */
  virtual std::unique_ptr<Model> parse(const nlohmann::json &data) const = 0;

  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes.
   *
   * This allows safe polymorphic destruction of parser objects when
   * accessed through a pointer to this base class.
   */
  virtual ~DataParser() = default;
};