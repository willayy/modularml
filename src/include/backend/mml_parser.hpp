#pragma once

#include "a_data_parser.hpp"

/**
 * @class Parser_mml
 * @brief A class for parsing JSON data of a model into a Model_mml object.
 */
class Parser_mml : public DataParser {
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
  std::unique_ptr<Model> parse(const nlohmann::json &data) const override;
};
