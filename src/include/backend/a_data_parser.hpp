#pragma once

#include "backend/a_model.hpp"
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

/**
 * @class DataParser
 * @brief Abstract class for parsing JSON data of a model into a Model Object.
 */
class DataParser {
public:
  /**
   * @brief Parses JSON data of a model into a Model object.
   *
   * @param data JSON data of a Model.
   */
  virtual std::unique_ptr<Model> parse(const nlohmann::json &data) const = 0;

  /// @brief Virtual destructor for cleanup.
  virtual ~DataParser() = default;
};