#pragma once
#include <concepts>
#include <memory>
#include <type_traits>

/**
 * @namespace TensorConcept
 * @brief Namespace containing concepts related to tensor operations
 *
 * This namespace defines concepts that constrain the types that can be used
 * with tensor operations in the ModularML framework.
 */
namespace TensorConcept {

/**
 * @concept Types
 * @brief Concept defining valid types for tensor elements
 *
 * This concept constrains tensor element types to arithmetic types (int, float,
 * double, etc.) to ensure that mathematical operations can be performed on
 * them.
 *
 * @tparam T The type to check against the concept
 */
template <typename T>
concept Types = std::is_arithmetic_v<T>;
}  // namespace TensorConcept
