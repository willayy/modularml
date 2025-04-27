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

#include "datastructures/a_tensor.hpp"
#include "datastructures/mml_tensor.hpp"
#include "nodes/node_utils.hpp"
#include "operations/tensor_operations_module.hpp"

/**
 * @struct is_in_variant
 * @brief Type trait to check if a type is contained in a variant.
 *
 * @tparam T The type to check
 * @tparam Variant The variant type to check against
 */
template <typename T, typename Variant>
struct is_in_variant;

/**
 * @struct is_in_variant
 * @brief Specialization for std::variant types using a fold expression.
 *
 * @tparam T The type to check
 * @tparam Ts The types in the variant
 */
template <typename T, typename... Ts>
struct is_in_variant<T, std::variant<Ts...>>
    : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

/**
 * @brief Helper variable template for checking if a type is in a variant.
 *
 * @tparam T The type to check
 * @tparam Variant The variant type to check against
 */
template <typename T, typename Variant>
constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

/**
 * @struct TensorVariantMaker
 * @brief Template struct for creating tensor variants.
 *
 * @tparam Variant The variant type to transform
 */
template <typename Variant>
struct TensorVariantMaker;

/**
 * @struct TensorVariantMaker
 * @brief Specialization for creating variants of tensor shared pointers.
 *
 * @tparam Ts The types in the original variant
 */
template <typename... Ts>
struct TensorVariantMaker<std::variant<Ts...>> {
  using type = std::variant<std::shared_ptr<Tensor<Ts>>...>;
};

/**
 * @brief Helper type alias for creating tensor variants.
 *
 * @tparam Variant The variant type to transform
 */
template <typename Variant>
using TensorVariant = typename TensorVariantMaker<Variant>::type;

/**
 * @typedef GeneralDataTypes
 * @brief Type definition for supported tensor data types in the framework.
 *
 * This variant contains shared pointers to tensors of various numeric types.
 * Note: bfloat16 and float16 are not included as they're not native to C++17.
 */
using GeneralDataTypes = std::variant<
    std::shared_ptr<Tensor<bool>>, std::shared_ptr<Tensor<double>>,
    std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<int16_t>>,
    std::shared_ptr<Tensor<int32_t>>, std::shared_ptr<Tensor<int64_t>>,
    std::shared_ptr<Tensor<int8_t>>, std::shared_ptr<Tensor<uint16_t>>,
    std::shared_ptr<Tensor<uint32_t>>,
    std::shared_ptr<Tensor<uint64_t>>,  // unsigned long long or uint64_t
    std::shared_ptr<Tensor<uint8_t>>>;

/**
 * @class Node
 * @brief Abstract base class representing a node in a computational graph.
 *
 * This class defines the interface for nodes in a computational graph,
 * including methods for forward propagation, input/output management,
 * and checking the status of inputs and outputs.
 */
class Node {
 public:
  /**
   * @brief Perform the forward pass computation.
   *
   * This pure virtual function must be overridden by derived classes to
   * implement the specific forward pass logic. It modifies the output(s)
   * in place.
   *
   * @param iomap Map containing input and output tensors indexed by name
   */
  virtual void forward(
      std::unordered_map<std::string, GeneralDataTypes> &iomap) = 0;

  /**
   * @brief Get the names of input tensors required by this node.
   *
   * This pure virtual function must be overridden by derived classes to
   * provide the names of input tensors that the node expects.
   *
   * @return Vector of strings containing the names of the inputs to the node
   */
  virtual std::vector<std::string> getInputs() = 0;

  /**
   * @brief Get the names of output tensors produced by this node.
   *
   * This pure virtual function must be overridden by derived classes to
   * provide the names of output tensors that the node produces.
   *
   * @return Vector of strings containing the names of the outputs from the node
   */
  virtual std::vector<std::string> getOutputs() = 0;

  /**
   * @brief Virtual destructor for the Node class.
   *
   * Ensures derived class destructors are called properly when a Node is
   * deleted through a pointer to the base class.
   */
  virtual ~Node() = default;
};
