#pragma once

#include "a_data_parser.hpp"
#include "nlohmann/json_fwd.hpp"

/**
 * @class Parser_mml
 * @brief Concrete parser implementation for ModularML model definitions.
 *
 * This class implements the DataParser interface to parse ModularML-specific
 * model definitions from JSON format. It handles all aspects of converting
 * the serialized model representation into an executable Model_mml object,
 * including constructing the appropriate nodes and establishing their
 * connections.
 */
class Parser_mml : public DataParser {
 public:
  /**
   * @brief Default constructor for Parser_mml.
   *
   * Initializes a parser with no special configuration.
   */
  Parser_mml() = default;

  /**
   * @brief Parses JSON model definition into a Model_mml object.
   *
   * This method interprets the JSON data according to the ModularML model
   * format, extracting node definitions, tensor information, and connectivity
   * details. It constructs the appropriate computational graph structure,
   * creates the necessary nodes, and initializes tensor values as specified in
   * the definition.
   *
   * The returned Model object is actually a Model_mml instance accessed through
   * the base class interface.
   *
   * @param data JSON data containing the ModularML model definition
   * @return A unique pointer to the constructed Model_mml object
   */
  std::unique_ptr<Model> parse(const nlohmann::json &data) const override;
};
