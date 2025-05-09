#pragma once

#include "backend/model.hpp"
#include "nlohmann/json_fwd.hpp"

/**
 * @class Parser_mml
 * @brief A class for parsing JSON data of a model into a Model_mml object.
 */
class DataParser {
 public:
  DataParser() = delete;  // Prevent instantiation of this class

  /**
   * @brief Parses JSON data of a model into a Model_mml object.
   *
   * @param data JSON data of a Model.
   * @return The default representation of a model: Model_mml.
   */
  static std::unique_ptr<Model> parse(const nlohmann::json &data);
};
