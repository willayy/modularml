#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "backend/a_model.hpp"

/**
 * @namespace DataParser
 * @brief Namespace for parsing JSON data of a model into a Model_mml object.
 */
namespace DataParser {
  /**
   * @brief Parses JSON data of a model into a Model_mml object.
   *
   * @param data JSON data of a Model.
  std::unique_ptr<Model> parse(const nlohmann::json &data);
   */
  std::unique_ptr<Model> parse(const nlohmann::json &data);
};
