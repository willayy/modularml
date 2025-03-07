#pragma once

#include "a_model.hpp"
#include "globals.hpp"

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
  virtual unique_ptr<Model> parse(const json& data) const = 0;

  /// @brief Virtual destructor for cleanup.
  virtual ~DataParser() = default;
};